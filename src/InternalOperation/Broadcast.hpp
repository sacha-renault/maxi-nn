// #pragma once
// #include <vector>
// #include <eigen3/Eigen/Dense>

// namespace nn::Operation
// {
//     inline bool areShapeEqual(const std::vector<std::size_t>& s1, const std::vector<std::size_t>& s2) {
//         if (s1.size() != s2.size()) {
//             return false;
//         } else {
//             for (int i = 0 ; i < s1.size() ; ++i) {
//                 if (s1[i] != s2[i]) {
//                     return false;
//                 }
//             }
//         }
//         return true; // Broadcastable
//     }

//     inline bool areShapeBroadcastable(const std::vector<std::size_t>& s1, const std::vector<std::size_t>& s2) {
//         int len1 = s1.size();
//         int len2 = s2.size();

//         int maxLen = std::max(len1, len2);

//         for (int i = 0; i < maxLen; ++i) {
//             int dim1 = (i < len1) ? s1[len1 - 1 - i] : 1;
//             int dim2 = (i < len2) ? s2[len2 - 1 - i] : 1;

//             if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
//                 return false; // Not broadcastable
//             }
//         }

//         return true; // Broadcastable
//     }

//     inline std::vector<std::size_t> calculateBroadcastedShape(const std::vector<std::size_t>& s1, const std::vector<std::size_t>& s2) {
//         int i = s1.size() - 1;
//         int j = s2.size() - 1;
//         std::vector<std::size_t> result;

//         while (i >= 0 || j >= 0) {
//             int dim1 = (i >= 0) ? s1[i] : 1;
//             int dim2 = (j >= 0) ? s2[j] : 1;

//             if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
//                 return {}; // Return an empty vector indicating the shapes are not broadcastable
//             }

//             result.push_back(std::max(dim1, dim2));
//             i--;
//             j--;
//         }

//         std::reverse(result.begin(), result.end());
//         return result; // Broadcasted shape
//     }

// template <typename T>
// Eigen::Matrix<T, -1, -1> broadcastView(const Eigen::Matrix<T, -1, -1>& data,
//                                         const std::vector<std::size_t>& current_shape,
//                                         const std::vector<std::size_t>& target_shape) {
//     int current_rows = current_shape[0];
//     int target_rows = target_shape[0];
//     int data_size = data.size();

//     // Create a broadcasted matrix view
//     Eigen::Matrix<T, -1, -1> broadcasted(target_rows, data_size);

//     // Fill the broadcasted matrix by repeating the original data
//     for (int i = 0; i < target_rows; ++i) {
//         broadcasted.row(i) = data.row(0);
//     }

//     return broadcasted;
// }
// } // namespace nn::Operation
