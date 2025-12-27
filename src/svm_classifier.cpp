#include "utils.h"
#include "data_loader.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>

// ============================================================================
// SVM Classifier using LIBSVM format
// This implementation creates LIBSVM format files and uses system calls
// For direct integration, include svm.h from LIBSVM
// ============================================================================

class SVMClassifier {
private:
    std::string model_path;
    std::string model_name;
    double C;
    double gamma;
    bool trained;
    
    // Store predictions and stats
    std::vector<int> predictions;
    double accuracy;
    std::vector<std::vector<int>> confusion_matrix;
    
public:
    SVMClassifier(double c_param = 10.0, double gamma_param = -1.0, const std::string& name = "default")
        : model_name(name), C(c_param), gamma(gamma_param), trained(false), accuracy(0.0) {
        confusion_matrix.resize(Constants::CIFAR_CLASSES, 
                               std::vector<int>(Constants::CIFAR_CLASSES, 0));
    }
    
    // Convert features to LIBSVM format and save to file
    bool save_features_libsvm_format(const float* features, const int* labels,
                                     int num_samples, int feature_dim,
                                     const std::string& filepath) {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filepath << std::endl;
            return false;
        }
        
        for (int i = 0; i < num_samples; ++i) {
            // Write label
            file << labels[i];
            
            // Write features in sparse format (index:value)
            for (int j = 0; j < feature_dim; ++j) {
                float val = features[i * feature_dim + j];
                if (val != 0.0f) {
                    file << " " << (j + 1) << ":" << std::setprecision(6) << val;
                }
            }
            file << "\n";
        }
        
        file.close();
        return true;
    }
    
    // Train SVM using LIBSVM command line tool
    bool train(const float* features, const int* labels, int num_samples, int feature_dim) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Training SVM Classifier" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "C: " << C << std::endl;
        std::cout << "Gamma: " << (gamma < 0 ? "auto (1/num_features)" : std::to_string(gamma)) << std::endl;
        std::cout << "Samples: " << num_samples << std::endl;
        std::cout << "Features: " << feature_dim << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        Timer timer("SVM Training");
        
        // Calculate auto gamma if needed
        double actual_gamma = gamma < 0 ? (1.0 / feature_dim) : gamma;
        
        // Save features in LIBSVM format (unique per model variant)
        std::string train_file = "models/svm_train_" + model_name + ".txt";
        std::cout << "Saving training data to LIBSVM format..." << std::endl;
        if (!save_features_libsvm_format(features, labels, num_samples, feature_dim, train_file)) {
            return false;
        }
        
        // Build LIBSVM command (unique model path per variant)
        model_path = "models/svm_model_" + model_name + ".txt";
        std::stringstream cmd;
        cmd << "cd libsvm && ./svm-train "
            << "-s 0 "  // C-SVC
            << "-t 2 "  // RBF kernel (higher accuracy, slower training)
            << "-g " << actual_gamma << " "  // Gamma for RBF
            << "-c " << C << " "
            // No -q flag: show progress (dots for each 1000 iterations)
            << "../" << train_file << " "
            << "../" << model_path;
        
        std::cout << "Training SVM with RBF kernel (gamma=" << actual_gamma << ")..." << std::endl;
        std::cout << "WARNING: This may take several hours for 50k samples!" << std::endl;
        std::cout << "Progress: each line = one of 45 binary classifiers" << std::endl;
        std::cout << "Command: " << cmd.str() << std::endl;
        std::cout << std::endl;
        
        int result = system(cmd.str().c_str());
        
        if (result != 0) {
            std::cerr << "Error: SVM training failed" << std::endl;
            std::cerr << "Make sure LIBSVM is compiled: cd libsvm && make" << std::endl;
            return false;
        }
        
        trained = true;
        timer.print("Training completed");
        
        return true;
    }
    
    // Predict using trained SVM model
    bool predict(const float* features, int num_samples, int feature_dim,
                 std::vector<int>& pred_labels) {
        if (!trained) {
            std::cerr << "Error: SVM model not trained" << std::endl;
            return false;
        }
        
        Timer timer("SVM Prediction");
        
        // Save test features (unique per model variant)
        std::string test_file = "models/svm_test_" + model_name + ".txt";
        std::vector<int> dummy_labels(num_samples, 0);
        if (!save_features_libsvm_format(features, dummy_labels.data(), num_samples, feature_dim, test_file)) {
            return false;
        }
        
        // Predict using LIBSVM (unique output file per variant)
        std::string output_file = "models/svm_predictions_" + model_name + ".txt";
        std::stringstream cmd;
        cmd << "cd libsvm && ./svm-predict "
            << "-q "
            << "../" << test_file << " "
            << "../" << model_path << " "
            << "../" << output_file;
        
        int result = system(cmd.str().c_str());
        
        if (result != 0) {
            std::cerr << "Error: SVM prediction failed" << std::endl;
            return false;
        }
        
        // Read predictions
        std::ifstream pred_file(output_file);
        if (!pred_file.is_open()) {
            std::cerr << "Error: Cannot read predictions file" << std::endl;
            return false;
        }
        
        pred_labels.clear();
        int label;
        while (pred_file >> label) {
            pred_labels.push_back(label);
        }
        pred_file.close();
        
        timer.print("Prediction completed");
        
        return true;
    }
    
    // Evaluate predictions
    void evaluate(const std::vector<int>& predictions, const int* true_labels, int num_samples) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Evaluation Results" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        // Reset confusion matrix
        for (int i = 0; i < Constants::CIFAR_CLASSES; ++i) {
            for (int j = 0; j < Constants::CIFAR_CLASSES; ++j) {
                confusion_matrix[i][j] = 0;
            }
        }
        
        // Calculate accuracy and confusion matrix
        int correct = 0;
        for (int i = 0; i < num_samples; ++i) {
            int pred = predictions[i];
            int truth = true_labels[i];
            
            if (pred == truth) {
                correct++;
            }
            
            confusion_matrix[truth][pred]++;
        }
        
        accuracy = static_cast<double>(correct) / num_samples * 100.0;
        
        std::cout << "Overall Accuracy: " << std::fixed << std::setprecision(2) 
                  << accuracy << "% (" << correct << "/" << num_samples << ")" << std::endl;
        
        // Per-class accuracy
        std::cout << "\nPer-class Accuracy:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        for (int c = 0; c < Constants::CIFAR_CLASSES; ++c) {
            int class_total = 0;
            int class_correct = confusion_matrix[c][c];
            for (int j = 0; j < Constants::CIFAR_CLASSES; ++j) {
                class_total += confusion_matrix[c][j];
            }
            double class_acc = class_total > 0 ? 
                              (static_cast<double>(class_correct) / class_total * 100.0) : 0.0;
            std::cout << std::setw(12) << CIFAR10Dataset::get_class_name(c) 
                      << ": " << std::setw(6) << std::fixed << std::setprecision(2) 
                      << class_acc << "%" << std::endl;
        }
        
        // Print confusion matrix
        std::cout << "\nConfusion Matrix:" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "           ";
        for (int j = 0; j < Constants::CIFAR_CLASSES; ++j) {
            std::cout << std::setw(5) << j;
        }
        std::cout << std::endl;
        
        for (int i = 0; i < Constants::CIFAR_CLASSES; ++i) {
            std::cout << std::setw(10) << CIFAR10Dataset::get_class_name(i) << " ";
            for (int j = 0; j < Constants::CIFAR_CLASSES; ++j) {
                std::cout << std::setw(5) << confusion_matrix[i][j];
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
    }
    
    double get_accuracy() const { return accuracy; }
    const std::vector<std::vector<int>>& get_confusion_matrix() const { return confusion_matrix; }
};

// ============================================================================
// Standalone SVM Functions (for use in main.cpp)
// ============================================================================

void train_and_evaluate_svm(CIFAR10Dataset& dataset,
                            const std::string& train_features_path,
                            const std::string& test_features_path,
                            const std::string& model_name) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "SVM Training and Evaluation Pipeline" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model variant: " << model_name << std::endl;
    std::cout << "Training samples: " << Constants::CIFAR_TRAIN_SIZE << std::endl;
    std::cout << "Test samples: " << Constants::CIFAR_TEST_SIZE << std::endl;
    std::cout << "Feature dimension: " << Constants::FEATURE_DIM << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    Timer total_pipeline_timer("SVM Pipeline");
    
    // Load features from binary files
    float* train_features = new float[Constants::CIFAR_TRAIN_SIZE * Constants::FEATURE_DIM];
    float* test_features = new float[Constants::CIFAR_TEST_SIZE * Constants::FEATURE_DIM];
    
    std::ifstream train_file(train_features_path, std::ios::binary);
    if (!train_file.is_open()) {
        std::cerr << "Error: Cannot open train features file: " << train_features_path << std::endl;
        delete[] train_features;
        delete[] test_features;
        return;
    }
    train_file.read(reinterpret_cast<char*>(train_features),
                    Constants::CIFAR_TRAIN_SIZE * Constants::FEATURE_DIM * sizeof(float));
    train_file.close();
    
    std::ifstream test_file(test_features_path, std::ios::binary);
    if (!test_file.is_open()) {
        std::cerr << "Error: Cannot open test features file: " << test_features_path << std::endl;
        delete[] train_features;
        delete[] test_features;
        return;
    }
    test_file.read(reinterpret_cast<char*>(test_features),
                   Constants::CIFAR_TEST_SIZE * Constants::FEATURE_DIM * sizeof(float));
    test_file.close();
    
    std::cout << "Features loaded successfully." << std::endl;
    
    // Create SVM classifier with unique model name
    SVMClassifier svm(10.0, -1.0, model_name);  // C=10, gamma=auto, unique name
    
    // Train SVM
    Timer train_timer("SVM Training");
    if (!svm.train(train_features, dataset.get_train_labels(),
                   Constants::CIFAR_TRAIN_SIZE, Constants::FEATURE_DIM)) {
        std::cerr << "SVM training failed" << std::endl;
        delete[] train_features;
        delete[] test_features;
        return;
    }
    double train_time = train_timer.elapsed();
    std::cout << "SVM training completed in " << std::fixed << std::setprecision(2) << train_time << "s" << std::endl;
    
    // Predict on test set
    Timer predict_timer("SVM Prediction");
    std::vector<int> predictions;
    if (!svm.predict(test_features, Constants::CIFAR_TEST_SIZE, Constants::FEATURE_DIM, predictions)) {
        std::cerr << "SVM prediction failed" << std::endl;
        delete[] train_features;
        delete[] test_features;
        return;
    }
    double predict_time = predict_timer.elapsed();
    std::cout << "SVM prediction completed in " << std::fixed << std::setprecision(2) << predict_time << "s" << std::endl;
    
    // Evaluate
    Timer eval_timer("Evaluation");
    svm.evaluate(predictions, dataset.get_test_labels(), Constants::CIFAR_TEST_SIZE);
    double eval_time = eval_timer.elapsed();
    
    double total_pipeline_time = total_pipeline_timer.elapsed();
    
    // Clean up
    delete[] train_features;
    delete[] test_features;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "SVM Pipeline Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "SVM training time:   " << std::fixed << std::setprecision(2) << train_time << "s" << std::endl;
    std::cout << "SVM prediction time: " << predict_time << "s" << std::endl;
    std::cout << "Evaluation time:     " << eval_time << "s" << std::endl;
    std::cout << "Total pipeline time: " << total_pipeline_time << "s" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Test Accuracy:       " << svm.get_accuracy() << "%" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// Scale features for better SVM performance
void scale_features(float* features, int num_samples, int feature_dim) {
    // Find min and max for each feature
    std::vector<float> min_vals(feature_dim, 1e38f);
    std::vector<float> max_vals(feature_dim, -1e38f);
    
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            float val = features[i * feature_dim + j];
            min_vals[j] = std::min(min_vals[j], val);
            max_vals[j] = std::max(max_vals[j], val);
        }
    }
    
    // Scale to [0, 1]
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < feature_dim; ++j) {
            float range = max_vals[j] - min_vals[j];
            if (range > 1e-10f) {
                features[i * feature_dim + j] = 
                    (features[i * feature_dim + j] - min_vals[j]) / range;
            } else {
                features[i * feature_dim + j] = 0.0f;
            }
        }
    }
}


