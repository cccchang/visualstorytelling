package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//im2story/...",
    ],
)

py_binary(
    name = "build_vist_data",
    srcs = [
        "data/build_vist_data.py",
    ],
)

sh_binary(
    name = "preprocess_vist",
    srcs = ["data/preprocess_vist.sh"],
    data = [
        ":build_vist_data",
    ],
)

py_library(
    name = "configuration",
    srcs = ["configuration.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "visual_storytelling_model",
    srcs = ["visual_storytelling_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//im2story/ops:image_embedding",
        "//im2story/ops:image_processing",
        "//im2story/ops:inputs",
    ],
)

py_test(
    name = "visual_storytelling_model_test",
    size = "large",
    srcs = ["visual_storytelling_model_test.py"],
    deps = [
        ":configuration",
        ":visual_storytelling_model",
    ],
)

py_library(
    name = "inference_wrapper",
    srcs = ["inference_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":visual_storytelling_model",
        "//im2story/inference_utils:inference_wrapper_base",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":visual_storytelling_model",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":visual_storytelling_model",
    ],
)

py_binary(
    name = "run_inference",
    srcs = ["run_inference.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":inference_wrapper",
        "//im2story/inference_utils:caption_generator",
        "//im2story/inference_utils:vocabulary",
    ],
)
