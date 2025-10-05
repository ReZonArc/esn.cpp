#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-model.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

int main() {
    std::cout << "Testing ESN architecture support...\n";

    // Test 1: Verify ESN architecture is recognized
    {
        std::cout << "Test 1: ESN architecture recognition... ";
        
        // Test architecture name mapping
        llm_arch esn_arch = llm_arch_from_string("esn");
        assert(esn_arch == LLM_ARCH_ESN);
        
        const char* arch_name = llm_arch_name(esn_arch);
        assert(std::string(arch_name) == "esn");
        
        std::cout << "PASSED\n";
    }

    // Test 2: Verify ESN is recognized as recurrent
    {
        std::cout << "Test 2: ESN recurrent architecture... ";
        
        bool is_recurrent = llm_arch_is_recurrent(LLM_ARCH_ESN);
        assert(is_recurrent == true);
        
        std::cout << "PASSED\n";
    }

    // Test 3: Verify ESN tensor info mappings exist
    {
        std::cout << "Test 3: ESN tensor info mappings... ";
        
        // Test that ESN tensor infos are properly defined
        const llm_tensor_info & input_info = llm_tensor_info_for(LLM_TENSOR_ESN_INPUT_WEIGHTS);
        assert(input_info.layer == LLM_TENSOR_LAYER_INPUT);
        assert(input_info.op == GGML_OP_MUL_MAT);
        
        const llm_tensor_info & reservoir_info = llm_tensor_info_for(LLM_TENSOR_ESN_RESERVOIR_WEIGHTS);
        assert(reservoir_info.layer == LLM_TENSOR_LAYER_REPEATING);
        assert(reservoir_info.op == GGML_OP_MUL_MAT);
        
        const llm_tensor_info & output_info = llm_tensor_info_for(LLM_TENSOR_ESN_OUTPUT_WEIGHTS);
        assert(output_info.layer == LLM_TENSOR_LAYER_OUTPUT);
        assert(output_info.op == GGML_OP_MUL_MAT);
        
        std::cout << "PASSED\n";
    }

    // Test 4: Test ESN hyperparameter initialization
    {
        std::cout << "Test 4: ESN hyperparameter defaults... ";
        
        llama_hparams hparams;
        
        // Verify default ESN parameter values
        assert(hparams.esn_reservoir_size == 0);       // Should be set when loading model
        assert(hparams.esn_spectral_radius == 0.95f);  // Conservative default
        assert(hparams.esn_sparsity == 0.1f);          // 10% connectivity  
        assert(hparams.esn_leaking_rate == 1.0f);      // No leaky integration by default
        assert(hparams.esn_input_scaling == 1.0f);     // No scaling by default
        
        std::cout << "PASSED\n";
    }

    // Test 5: Test ESN tensor name generation
    {
        std::cout << "Test 5: ESN tensor name generation... ";
        
        const auto tn = LLM_TN(LLM_ARCH_ESN);
        
        std::string input_name = tn(LLM_TENSOR_ESN_INPUT_WEIGHTS);
        assert(input_name == "esn_input_weights");
        
        std::string reservoir_name = tn(LLM_TENSOR_ESN_RESERVOIR_WEIGHTS);
        assert(reservoir_name == "esn_reservoir_weights");
        
        std::string output_name = tn(LLM_TENSOR_ESN_OUTPUT_WEIGHTS);
        assert(output_name == "esn_output_weights");
        
        std::cout << "PASSED\n";
    }

    std::cout << "\nAll ESN tests passed successfully!\n";
    
    // Print some information about ESN capabilities
    std::cout << "\nESN Architecture Features:\n";
    std::cout << "- Recurrent architecture: " << (llm_arch_is_recurrent(LLM_ARCH_ESN) ? "Yes" : "No") << "\n";
    std::cout << "- Hybrid architecture: " << (llm_arch_is_hybrid(LLM_ARCH_ESN) ? "Yes" : "No") << "\n";
    std::cout << "- Diffusion architecture: " << (llm_arch_is_diffusion(LLM_ARCH_ESN) ? "Yes" : "No") << "\n";
    std::cout << "- Architecture name: " << llm_arch_name(LLM_ARCH_ESN) << "\n";

    return EXIT_SUCCESS;
}