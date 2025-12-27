#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "utils.h"
#include <string>
#include <vector>

// CIFAR-10 Dataset class
class CIFAR10Dataset {
private:
    std::vector<float> train_images;  // [50000, 3, 32, 32] normalized to [0,1]
    std::vector<float> test_images;   // [10000, 3, 32, 32] normalized to [0,1]
    std::vector<int> train_labels;    // [50000]
    std::vector<int> test_labels;     // [10000]
    
    bool loaded;
    std::string data_path;
    
    // Helper function to load a single batch file
    bool load_batch_file(const std::string& filename, 
                         std::vector<float>& images, 
                         std::vector<int>& labels,
                         int start_idx);
    
public:
    CIFAR10Dataset();
    ~CIFAR10Dataset();
    
    // Load dataset from directory
    bool load(const std::string& path);
    
    // Download dataset if not present (requires network)
    bool download(const std::string& path);
    
    // Getters
    const float* get_train_images() const { return train_images.data(); }
    const float* get_test_images() const { return test_images.data(); }
    const int* get_train_labels() const { return train_labels.data(); }
    const int* get_test_labels() const { return test_labels.data(); }
    
    int get_train_size() const { return Constants::CIFAR_TRAIN_SIZE; }
    int get_test_size() const { return Constants::CIFAR_TEST_SIZE; }
    
    // Get single image (for visualization)
    void get_image(float* dst, int index, bool is_train = true) const;
    
    // Get batch of images
    void get_batch(float* dst, const std::vector<int>& indices, bool is_train = true) const;
    
    // Get batch labels
    void get_batch_labels(int* dst, const std::vector<int>& indices, bool is_train = true) const;
    
    // Check if loaded
    bool is_loaded() const { return loaded; }
    
    // Class names
    static const char* get_class_name(int label);
};

// Batch generator for training
class BatchGenerator {
private:
    const CIFAR10Dataset& dataset;
    std::vector<int> indices;
    int batch_size;
    int current_idx;
    bool is_train;
    RandomGenerator rng;
    
public:
    BatchGenerator(const CIFAR10Dataset& ds, int batch_sz, bool train = true, unsigned int seed = 42);
    
    // Reset and shuffle for new epoch
    void reset(bool shuffle = true);
    
    // Check if more batches available
    bool has_next() const;
    
    // Get next batch (returns actual batch size, may be smaller for last batch)
    int next_batch(float* images, int* labels = nullptr);
    
    // Get current batch indices
    void get_current_indices(std::vector<int>& batch_indices) const;
    
    // Get total number of batches
    int num_batches() const;
};

#endif // DATA_LOADER_H


