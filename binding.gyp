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
        "$(CUDA_PATH)/include"
      ],
      "defines": [
        "NAPI_DISABLE_CPP_EXCEPTIONS"
      ],
      "conditions": [
        ["OS=='win'", {
          "libraries": [
            "-L$(CUDA_PATH)/lib/x64",
            "-lcudart",
            "-lcurand"
          ],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "AdditionalOptions": ["/std:c++17"]
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
        }]
      ],
      "rules": [
        {
          "extension": "cu",
          "inputs": ["<(RULE_INPUT_PATH)"],
          "outputs": ["<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o"],
          "rule_name": "cuda_compile",
          "process_outputs_as_sources": 1,
          "conditions": [
            ["OS=='win'", {
              "action": [
                "nvcc",
                "-c",
                "-o", "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o",
                "<(RULE_INPUT_PATH)",
                "-std=c++14",
                "-Xcompiler", "/EHsc,/W3,/nologo,/Od,/Zi",
                "--expt-relaxed-constexpr",
                "-I", "<!@(node -p \"require('node-addon-api').include.replace(/\\\\\\\\/g, '/')\")",
                "-I", "<(module_root_dir)"
              ]
            }],
            ["OS=='linux'", {
              "action": [
                "nvcc",
                "-c",
                "-o", "<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).o",
                "<(RULE_INPUT_PATH)",
                "-std=c++14",
                "-Xcompiler", "-fPIC,-O2",
                "--expt-relaxed-constexpr",
                "-I", "<!@(node -p \"require('node-addon-api').include.replace(/\\\\\\\\/g, '/')\")",
                "-I", "<(module_root_dir)"
              ]
            }]
          ],
          "message": "Compiling CUDA source <(RULE_INPUT_PATH)"
        }
      ]
    }
  ]
}
