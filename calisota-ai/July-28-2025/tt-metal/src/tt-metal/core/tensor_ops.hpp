/**
 * @file tensor_ops.hpp
 * @brief Tensor operations optimized for TT-Metal
 *
 * Implements population geometry transformations, matrix operations,
 * and neural activity manipulations using Tensix cores.
 *
 * @author GRYPHGEN Project
 * @date 2025
 * @license Apache 2.0
 */

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>

namespace dynamic_cortex {
namespace tt_metal_backend {

/**
 * @brief Tensor shape descriptor
 */
struct TensorShape {
    uint32_t rows;
    uint32_t cols;
    uint32_t depth;
    uint32_t batch;

    TensorShape() : rows(1), cols(1), depth(1), batch(1) {}
    TensorShape(uint32_t r, uint32_t c) : rows(r), cols(c), depth(1), batch(1) {}
    TensorShape(uint32_t r, uint32_t c, uint32_t d) : rows(r), cols(c), depth(d), batch(1) {}
    TensorShape(uint32_t r, uint32_t c, uint32_t d, uint32_t b)
        : rows(r), cols(c), depth(d), batch(b) {}

    size_t size() const { return rows * cols * depth * batch; }
    size_t spatial_size() const { return rows * cols; }
};

/**
 * @brief Math fidelity levels for Tensix operations
 */
enum class MathFidelity {
    LoFi,       ///< Low fidelity (faster, less accurate)
    HiFi2,      ///< Medium fidelity
    HiFi3,      ///< High fidelity
    HiFi4       ///< Highest fidelity (slower, most accurate)
};

/**
 * @brief Tile layout modes
 */
enum class TileLayout {
    ROW_MAJOR,     ///< Standard row-major layout
    COL_MAJOR,     ///< Column-major layout
    TILED_32x32,   ///< 32x32 tiled layout (optimized for Tensix)
    CUSTOM         ///< Custom layout
};

/**
 * @brief Tensor descriptor for TT-Metal
 */
class TTTensor {
public:
    TTTensor();
    TTTensor(const TensorShape& shape, TileLayout layout = TileLayout::ROW_MAJOR);
    ~TTTensor();

    // Shape and layout
    TensorShape shape() const { return shape_; }
    TileLayout layout() const { return layout_; }
    size_t size() const { return shape_.size(); }
    size_t sizeBytes() const { return size() * sizeof(float); }

    // Data access
    float* data();
    const float* data() const;

    // Memory management
    void allocate();
    void deallocate();
    bool isAllocated() const;

    // Device management
    void toDevice();
    void fromDevice();
    bool isOnDevice() const;

    // Operations (in-place)
    void fill(float value);
    void randomize(float min = 0.0f, float max = 1.0f);
    void zero();

private:
    TensorShape shape_;
    TileLayout layout_;
    float* host_data_;
    void* device_data_;  // Device buffer handle
    bool allocated_;
    bool on_device_;
};

/**
 * @brief Matrix multiplication operation
 */
class MatMulOp {
public:
    MatMulOp();

    // Configuration
    MatMulOp& setMathFidelity(MathFidelity fidelity);
    MatMulOp& setTransposeA(bool transpose);
    MatMulOp& setTransposeB(bool transpose);
    MatMulOp& setScalar(float alpha, float beta);  // C = alpha*A*B + beta*C
    MatMulOp& setTileLayout(TileLayout layout);

    // Execution
    void operator()(const TTTensor& A, const TTTensor& B, TTTensor& C);
    void execute(const TTTensor& A, const TTTensor& B, TTTensor& C);

    // Performance
    void enableProfiling(bool enable);
    float lastExecutionTimeMs() const;
    float gflopsPerSecond() const;

private:
    MathFidelity math_fidelity_;
    bool transpose_a_;
    bool transpose_b_;
    float alpha_;
    float beta_;
    TileLayout tile_layout_;
    bool profiling_enabled_;
    float last_execution_time_ms_;
};

/**
 * @brief Population covariance computation
 */
class CovarianceOp {
public:
    CovarianceOp();

    /**
     * @brief Compute covariance matrix of population activity
     * @param activity Input tensor (neurons x time)
     * @param covariance Output covariance matrix (neurons x neurons)
     */
    void operator()(const TTTensor& activity, TTTensor& covariance);

    // Options
    void setNormalized(bool normalized);  // Correlation matrix if true
    void setUnbiased(bool unbiased);      // Use N-1 denominator if true

private:
    bool normalized_;
    bool unbiased_;
    std::unique_ptr<MatMulOp> matmul_op_;
};

/**
 * @brief Principal Component Analysis
 */
class PCAOp {
public:
    PCAOp(uint32_t num_components);

    /**
     * @brief Compute PCA of population activity
     * @param activity Input tensor (neurons x time)
     * @param components Output principal components (neurons x num_components)
     * @param explained_variance Output variance explained by each PC
     */
    void operator()(const TTTensor& activity,
                    TTTensor& components,
                    std::vector<float>& explained_variance);

    void setNumComponents(uint32_t num_components);
    uint32_t numComponents() const { return num_components_; }

private:
    uint32_t num_components_;
    std::unique_ptr<CovarianceOp> cov_op_;

    void eigenDecomposition(const TTTensor& covariance,
                            TTTensor& eigenvectors,
                            std::vector<float>& eigenvalues);
};

/**
 * @brief Rotation matrix application
 */
class RotationOp {
public:
    RotationOp();

    /**
     * @brief Rotate population activity
     * @param activity Input/output tensor (neurons x time)
     * @param rotation_matrix Rotation matrix (neurons x neurons)
     * @param preserve_variance If true, normalize to preserve total variance
     */
    void operator()(TTTensor& activity,
                    const TTTensor& rotation_matrix,
                    bool preserve_variance = true);

    // Rotation matrix generation
    static void generateRotationMatrix(TTTensor& rotation_matrix, float angle_degrees);
    static void generateRandomRotation(TTTensor& rotation_matrix);

private:
    std::unique_ptr<MatMulOp> matmul_op_;

    float computeTotalVariance(const TTTensor& activity);
    void normalizeVariance(TTTensor& activity, float target_variance);
};

/**
 * @brief Variance normalization
 */
class VarianceNormalizationOp {
public:
    VarianceNormalizationOp();

    /**
     * @brief Normalize tensor to have target variance
     * @param tensor Input/output tensor
     * @param target_variance Desired total variance
     */
    void operator()(TTTensor& tensor, float target_variance);

    /**
     * @brief Compute current variance
     */
    float computeVariance(const TTTensor& tensor);

private:
    void computeMean(const TTTensor& tensor, float& mean);
    void scale(TTTensor& tensor, float scale_factor);
};

/**
 * @brief Sparse thresholding operation
 */
class SparseThresholdOp {
public:
    SparseThresholdOp(float threshold);

    /**
     * @brief Apply threshold and zero out values below it
     * @param tensor Input/output tensor
     * @return Number of elements zeroed
     */
    uint32_t operator()(TTTensor& tensor);

    void setThreshold(float threshold);
    float threshold() const { return threshold_; }

    // Statistics
    float sparsity() const;  // Fraction of zero elements
    uint32_t numNonZero() const;

private:
    float threshold_;
    uint32_t last_num_zeroed_;
    uint32_t last_total_elements_;
};

/**
 * @brief Element-wise operations
 */
class ElementwiseOp {
public:
    enum class Operation {
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE,
        MAX,
        MIN,
        RELU,
        TANH,
        SIGMOID
    };

    ElementwiseOp(Operation op);

    // Binary operations (A op B -> C)
    void operator()(const TTTensor& A, const TTTensor& B, TTTensor& C);

    // Unary operations (A op scalar -> B)
    void operator()(const TTTensor& A, float scalar, TTTensor& B);

    // Unary activation (A -> B)
    void operator()(const TTTensor& A, TTTensor& B);

private:
    Operation operation_;
};

/**
 * @brief Reduction operations
 */
class ReductionOp {
public:
    enum class Operation {
        SUM,
        MEAN,
        MAX,
        MIN,
        VARIANCE,
        STD_DEV
    };

    enum class Axis {
        ALL,      ///< Reduce entire tensor to scalar
        ROWS,     ///< Reduce along rows
        COLS,     ///< Reduce along columns
        DEPTH,    ///< Reduce along depth
        BATCH     ///< Reduce along batch
    };

    ReductionOp(Operation op, Axis axis);

    /**
     * @brief Perform reduction
     * @param input Input tensor
     * @param output Output tensor (reduced along specified axis)
     */
    void operator()(const TTTensor& input, TTTensor& output);

    /**
     * @brief Scalar reduction
     * @param input Input tensor
     * @return Scalar result
     */
    float reduce(const TTTensor& input);

private:
    Operation operation_;
    Axis axis_;
};

/**
 * @brief Feedforward transformation
 */
class FeedforwardTransform {
public:
    FeedforwardTransform(const TensorShape& input_shape,
                         const TensorShape& output_shape);

    /**
     * @brief Apply feedforward transformation V1 -> LM
     * @param v1_activity V1 population activity
     * @param lm_activity Output LM population activity
     * @param weights Connection weights (optional, uses learned if null)
     */
    void operator()(const TTTensor& v1_activity,
                    TTTensor& lm_activity,
                    const TTTensor* weights = nullptr);

    // Weight management
    void setWeights(const TTTensor& weights);
    const TTTensor& weights() const;
    void randomizeWeights(float mean = 0.0f, float std = 0.1f);

private:
    TensorShape input_shape_;
    TensorShape output_shape_;
    TTTensor weights_;
    std::unique_ptr<MatMulOp> matmul_op_;
};

/**
 * @brief Feedback transformation
 */
class FeedbackTransform {
public:
    FeedbackTransform(const TensorShape& lm_shape,
                      const TensorShape& v1_shape);

    /**
     * @brief Apply feedback transformation LM -> V1
     * @param lm_activity LM population activity
     * @param v1_activity Output V1 population activity (modulated)
     * @param context Behavioral context for modulation
     */
    void operator()(const TTTensor& lm_activity,
                    TTTensor& v1_activity,
                    bool is_rewarded);

    // Modulation strength
    void setModulationStrength(float strength);
    float modulationStrength() const { return modulation_strength_; }

    // Behavioral modulation
    void setRewardedStrength(float strength);
    void setNonRewardedStrength(float strength);

private:
    TensorShape lm_shape_;
    TensorShape v1_shape_;
    float modulation_strength_;
    float rewarded_strength_;
    float non_rewarded_strength_;
    TTTensor feedback_weights_;
    std::unique_ptr<MatMulOp> matmul_op_;
};

/**
 * @brief Utility functions for tensor operations
 */
namespace tensor_utils {

/**
 * @brief Copy tensor data
 */
void copy(const TTTensor& src, TTTensor& dst);

/**
 * @brief Transpose matrix
 */
void transpose(const TTTensor& input, TTTensor& output);

/**
 * @brief Reshape tensor
 */
void reshape(const TTTensor& input, TTTensor& output, const TensorShape& new_shape);

/**
 * @brief Concatenate tensors along axis
 */
void concatenate(const std::vector<TTTensor>& inputs, TTTensor& output, uint32_t axis);

/**
 * @brief Split tensor along axis
 */
void split(const TTTensor& input, std::vector<TTTensor>& outputs, uint32_t axis);

/**
 * @brief Compute L2 norm
 */
float l2_norm(const TTTensor& tensor);

/**
 * @brief Print tensor shape and statistics
 */
void print_stats(const TTTensor& tensor, const std::string& name = "");

} // namespace tensor_utils

} // namespace tt_metal_backend
} // namespace dynamic_cortex
