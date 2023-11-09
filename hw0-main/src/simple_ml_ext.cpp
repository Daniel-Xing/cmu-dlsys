#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

#include <cmath>
#include <vector>
#include <algorithm>

namespace py = pybind11;


// Helper function to compute the softmax probabilities for a vector
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0;

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] /= sum_exp;
    }

    return probabilities;
}

// Function to compute the gradient of the softmax loss
void softmax_grad(const std::vector<float>& Z, const std::vector<unsigned char>& y,
                  size_t batch_size, size_t num_classes, std::vector<float>& gradient) {
    std::vector<float> probabilities;

    // Compute softmax probabilities
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> logits(Z.begin() + i * num_classes, Z.begin() + (i + 1) * num_classes);
        std::vector<float> probs = softmax(logits);

        // Subtract 1 from the probability of the correct class
        probs[y[i]] -= 1;

        // Copy the probabilities back into the gradient vector
        std::copy(probs.begin(), probs.end(), gradient.begin() + i * num_classes);
    }

    // Average the gradient over the batch
    for (size_t i = 0; i < gradient.size(); ++i) {
        gradient[i] /= static_cast<float>(batch_size);
    }
}

void dot_product(const float* A, const float* B, float* C, size_t A_rows, size_t A_cols, size_t B_cols) {
    // Initialize C with zeros
    std::fill(C, C + A_rows * B_cols, 0.0f);

    // Compute the dot product
    for (size_t i = 0; i < A_rows; ++i) {            // Iterate over the rows of A
        for (size_t j = 0; j < B_cols; ++j) {        // Iterate over the columns of B
            for (size_t k = 0; k < A_cols; ++k) {    // Dot product calculation
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    // Allocate memory for logits and gradients
    std::vector<float> logits(batch * k);
    std::vector<float> gradients(batch * k);

    for (size_t i = 0; i < m; i += batch) {
        size_t current_batch_size = std::min(batch, m - i);

        // Compute logits for the current batch
        dot_product(X + i * n, theta, logits.data(), current_batch_size, n, k);

        // Convert the raw logits array into a std::vector<float> for softmax_grad function
        std::vector<float> logits_vector(logits.data(), logits.data() + current_batch_size * k);
        
        // Initialize the gradient vector to hold the gradients for current batch
        std::vector<float> gradient_vector(current_batch_size * k);

        // Now call the softmax_grad function
        softmax_grad(logits_vector, std::vector<unsigned char>(y + i, y + i + current_batch_size),
                     current_batch_size, k, gradient_vector);

        // Update theta based on gradients
        for (size_t j = 0; j < current_batch_size; ++j) {
            for (size_t c = 0; c < k; ++c) {
                for (size_t d = 0; d < n; ++d) {
                    theta[d * k + c] -= lr * gradient_vector[j * k + c] * X[(i + j) * n + d];
                }
            }
        }
    }
}



/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
