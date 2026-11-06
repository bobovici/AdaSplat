#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions
    m.def("nd_rasterize_forward", &nd_rasterize_forward_tensor,
      py::arg("tile_bounds"), py::arg("block"), py::arg("img_size"),
      py::arg("gaussian_ids_sorted"), py::arg("tile_bins"),
      py::arg("xys"), py::arg("conics"), py::arg("colors"),
      py::arg("opacities"), py::arg("background"),
      py::arg("p"));
m.def("nd_rasterize_backward", &nd_rasterize_backward_tensor,
      py::arg("img_height"), py::arg("img_width"),
      py::arg("BLOCK_H"), py::arg("BLOCK_W"),
      py::arg("gaussians_ids_sorted"), py::arg("tile_bins"),
      py::arg("xys"), py::arg("conics"), py::arg("colors"),
      py::arg("opacities"), py::arg("background"),
      py::arg("final_Ts"), py::arg("final_idx"),
      py::arg("v_output"), py::arg("v_output_alpha"),
      py::arg("p"));
m.def("rasterize_forward", &rasterize_forward_tensor,
      py::arg("tile_bounds"), py::arg("block"), py::arg("img_size"),
      py::arg("gaussian_ids_sorted"), py::arg("tile_bins"),
      py::arg("xys"), py::arg("conics"), py::arg("colors"),
      py::arg("opacities"), py::arg("background"),
      py::arg("p"));
m.def("rasterize_backward", &rasterize_backward_tensor,
      py::arg("img_height"), py::arg("img_width"),
      py::arg("BLOCK_H"), py::arg("BLOCK_W"),
      py::arg("gaussians_ids_sorted"), py::arg("tile_bins"),
      py::arg("xys"), py::arg("conics"), py::arg("colors"),
      py::arg("opacities"), py::arg("background"),
      py::arg("final_Ts"), py::arg("final_idx"),
      py::arg("v_output"), py::arg("v_output_alpha"),
      py::arg("p"));
m.def("rasterize_sum_forward", &rasterize_forward_sum_tensor,
      py::arg("tile_bounds"), py::arg("block"), py::arg("img_size"),
      py::arg("gaussian_ids_sorted"), py::arg("tile_bins"),
      py::arg("xys"), py::arg("conics"), py::arg("colors"),
      py::arg("opacities"), py::arg("background"),
      py::arg("p"));
m.def("rasterize_sum_backward", &rasterize_backward_sum_tensor,
      py::arg("img_height"), py::arg("img_width"),
      py::arg("BLOCK_H"), py::arg("BLOCK_W"),
      py::arg("gaussians_ids_sorted"), py::arg("tile_bins"),
      py::arg("xys"), py::arg("conics"), py::arg("colors"),
      py::arg("opacities"), py::arg("background"),
      py::arg("final_Ts"), py::arg("final_idx"),
      py::arg("v_output"), py::arg("v_output_alpha"),
      py::arg("p"));


    m.def("project_gaussians_forward", &project_gaussians_forward_tensor);
    m.def("project_gaussians_backward", &project_gaussians_backward_tensor);
    m.def("compute_sh_forward", &compute_sh_forward_tensor);
    m.def("compute_sh_backward", &compute_sh_backward_tensor);
    m.def("project_gaussians_2d_forward", &project_gaussians_2d_forward_tensor);
    m.def("project_gaussians_2d_backward", &project_gaussians_2d_backward_tensor);
    m.def("project_gaussians_2d_scale_rot_forward", &project_gaussians_2d_scale_rot_forward_tensor);
    m.def("project_gaussians_2d_scale_rot_backward", &project_gaussians_2d_scale_rot_backward_tensor);
    // utils
    m.def("compute_cov2d_bounds", &compute_cov2d_bounds_tensor);
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);


    m.def("nd_rasterize_lp_forward", &nd_rasterize_lp_forward_tensor,
      "ND LP Rasterize Forward",
      py::arg("tile_bounds"), py::arg("block"), py::arg("img_size"),
      py::arg("gaussian_ids_sorted"), py::arg("tile_bins"),
      py::arg("xys"), py::arg("scales"), py::arg("rotations"),
      py::arg("p"), py::arg("colors"), py::arg("opacities"),
      py::arg("background"), py::arg("alpha_cutoff") = 1.0 / 255.0,
      py::arg("p_min") = 1.05, py::arg("p_max") = 2.0);

     m.def("nd_rasterize_lp_backward", &nd_rasterize_lp_backward_tensor,
          "ND LP Rasterize Backward",
          py::arg("img_height"), py::arg("img_width"),
          py::arg("BLOCK_H"), py::arg("BLOCK_W"),
          py::arg("gaussian_ids_sorted"), py::arg("tile_bins"),
          py::arg("xys"), py::arg("scales"), py::arg("rotations"),
          py::arg("colors"), py::arg("opacities"),
          py::arg("background"), py::arg("final_Ts"), py::arg("final_idx"),
          py::arg("v_output"), py::arg("v_output_alpha"),
          py::arg("p"), py::arg("alpha_cutoff") = 1.0 / 255.0,
          py::arg("p_min") = 1.05, py::arg("p_max") = 2.0);

}


