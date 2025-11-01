#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

using json = nlohmann::json;

struct json_entry {
    uint32_t    uid;
    std::string question_text;
    std::string answer_text;
    std::string model_answer;
    std::string target;
};

/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
    uint32_t             uid;
    std::string          target_layer;
    std::string          output_dir;
};

static std::string ggml_type_to_numpy_descr(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return "<f4";
        case GGML_TYPE_F16:
            return "<f2";
        case GGML_TYPE_Q4_0:
            return "<i1";
        case GGML_TYPE_Q4_1:
            return "<i1";
        case GGML_TYPE_Q5_0:
            return "<i1";
        case GGML_TYPE_Q5_1:
            return "<i1";
        case GGML_TYPE_Q8_0:
            return "<i1";
        case GGML_TYPE_I8:
            return "<i1";
        case GGML_TYPE_I16:
            return "<i2";
        case GGML_TYPE_I32:
            return "<i4";
        case GGML_TYPE_I64:
            return "<i8";
        default:
            throw std::runtime_error("Unsupported ggml type");
    }
}

static void save_data_npy(const void *                 data,
                          size_t                       n_bytes,
                          const std::vector<int64_t> & shape,
                          ggml_type                    type,
                          const char *                 path) {
    std::ostringstream hdr;
    hdr << "{";
    hdr << "\"descr\": \"" << ggml_type_to_numpy_descr(type) << "\", ";
    hdr << "\"fortran_order\": False, ";
    hdr << "\"shape\": (";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            hdr << ", ";
        }
        hdr << shape[i];
    }
    if (shape.size() == 1) {
        hdr << ",";
    }
    hdr << ")}";

    std::string header = hdr.str();

    const size_t magic_len        = 6;  // "\x93NUMPY"
    const size_t version_len      = 2;  // major, minor
    const size_t header_len_field = 2;  // uint16 little‑endian

    size_t total_len = magic_len + version_len + header_len_field + header.size();
    size_t pad       = (64 - (total_len % 64)) % 64;
    header.append(pad, ' ');  // official padding character

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error(std::string("Cannot open file: ") + path);
    }

    // Magic + version 1.0
    out.write("\x93NUMPY", 6);
    out.put(1);  // major
    out.put(0);  // minor

    // Header length (little‑endian uint16)
    uint16_t hdr_len = static_cast<uint16_t>(header.size());
    out.put(static_cast<char>(hdr_len & 0xFF));
    out.put(static_cast<char>((hdr_len >> 8) & 0xFF));

    // Header bytes
    out.write(header.c_str(), header.size());

    // Raw data payload
    out.write(reinterpret_cast<const char *>(data), n_bytes);
}

static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true;
    }

    auto * cb_data = static_cast<callback_data *>(user_data);

    // Store only the tensor whose name matches the target layer
    if (std::string(t->name) == cb_data->target_layer) {
        const size_t n_bytes = ggml_nbytes(t);
        const bool   is_host = ggml_backend_buffer_is_host(t->buffer);

        const void * data_ptr = nullptr;
        if (!is_host) {
            cb_data->data.resize(n_bytes);
            ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
            data_ptr = cb_data->data.data();
        } else {
            data_ptr = static_cast<const void *>(t->data);
        }

        // Build shape vector (trim trailing zeros)
        std::vector<int64_t> shape;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            shape.push_back(t->ne[i]);
        }
        if (shape.empty()) {
            shape.push_back(1);
        }

        std::string out_path = cb_data->output_dir + '/' + std::to_string(cb_data->uid) + ".npy";
        save_data_npy(data_ptr, n_bytes, shape, t->type, out_path.c_str());
    }
    return true;
}

static bool run_one(llama_context * ctx, const common_params & params, const std::string & prompt) {
    const llama_model * model   = llama_get_model(ctx);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const bool          add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, add_bos);
    if (tokens.empty()) {
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }
    return true;
}

static std::vector<json_entry> load_input_json(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open INPUT_JSON: " + path);
    }

    json j;
    in >> j;

    std::vector<json_entry> entries;
    for (const auto & obj : j) {
        json_entry e;
        e.uid           = obj.at("uid").get<uint32_t>();
        e.question_text = obj.at("question_text").get<std::string>();
        e.answer_text   = obj.at("answer_text").get<std::string>();
        e.model_answer  = obj.at("model_answer").get<std::string>();
        e.target        = obj.at("target").get<std::string>();
        entries.push_back(std::move(e));
    }
    return entries;
}

int main(int argc, char ** argv) {
    // keep your original constants, but expand them to full paths
    const std::string TARGET_LAYER = std::string("l_out-15");
    const std::string INPUT_JSON   = std::string(std::getenv("HOME")) + "/310-solution/datasets/splits/train.json";
    const std::string OUTPUT_DIR   = std::string(std::getenv("HOME")) + "/310-solution/npy_outputs/";

    // make sure the output directory exists
    std::filesystem::create_directories(OUTPUT_DIR);

    callback_data cb_data{
        .data         = {},
        .uid          = 0,
        .target_layer = TARGET_LAYER,
        .output_dir   = OUTPUT_DIR,
    };

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

    common_init_result llama_init = common_init_from_params(params);
    llama_model *      model      = llama_init.model.get();
    llama_context *    ctx        = llama_init.context.get();
    if (!model || !ctx) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    std::vector<json_entry> entries = load_input_json(INPUT_JSON);
    for (const auto & e : entries) {
        cb_data.uid = e.uid;

        // run inference on the question text
        if (!run_one(ctx, params, e.question_text)) {
            LOG_ERR("Inference failed for uid %u\n", e.uid);
            continue;
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
