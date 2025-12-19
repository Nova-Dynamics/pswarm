{
  "targets": [
    {
      "target_name": "pswarm",
      "sources": [
        "src/pswarm_wrapper.cpp",
        "ParticleSlam.cu"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        ".",
        "/usr/local/cuda/include"
      ],
      "defines": [
        "NAPI_DISABLE_CPP_EXCEPTIONS"
      ],
      "conditions": [
        ["OS=='win'", {
          "actions": [{
            "action_name": "error_windows_not_supported",
            "inputs": [],
            "outputs": ["<(PRODUCT_DIR)/error.txt"],
            "action": ["echo", "ERROR: Windows is not supported. This project requires Linux with CUDA."]
          }],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "AdditionalOptions": ["/FORCE_ERROR_WINDOWS_NOT_SUPPORTED"]
            }
          }
        }],
        ["OS=='linux'", {
          "libraries": [
            "-L/usr/local/cuda/lib64",
            "-lcudart",
            "-lcurand"
          ],
          "cflags_cc": ["-std=c++14"],
          "ldflags": [
            "-Wl,-rpath,/usr/local/cuda/lib64"
          ]
        }],
        ["OS!='linux'", {
          "type": "none"
        }]
      ],
      "rules": [
        {
          "extension": "cu",
          "inputs": ["<(RULE_INPUT_PATH)"],
          "outputs": ["<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o"],
          "rule_name": "cuda_compile",
          "process_outputs_as_sources": 1,
          "action": [
            "sh",
            "<(module_root_dir)/scripts/nvcc-wrapper.sh",
            "-c",
            "-o", "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o",
            "<(RULE_INPUT_PATH)",
            "-std=c++14",
            "-Xcompiler", "-fPIC,-O2",
            "--expt-relaxed-constexpr",
            "--use_fast_math",
            "-I", "<!@(node -p \"require('node-addon-api').include.replace(/\\\\\\\\/g, '/')\")",
            "-I", "<(module_root_dir)"
          ],
          "message": "Compiling CUDA source <(RULE_INPUT_PATH)"
        }
      ]
    }
  ]
}
