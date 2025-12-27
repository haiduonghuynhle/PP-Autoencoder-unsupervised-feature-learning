#include "data_loader.h"
#include <cstdio>
#include <cstdlib>

// CIFAR-10 class names
static const char* CIFAR10_CLASSES[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

// ============================================================================
// CIFAR10Dataset Implementation
// ============================================================================

CIFAR10Dataset::CIFAR10Dataset() : loaded(false) {
    train_images.reserve(Constants::CIFAR_TRAIN_SIZE * Constants::CIFAR_IMG_PIXELS);
    test_images.reserve(Constants::CIFAR_TEST_SIZE * Constants::CIFAR_IMG_PIXELS);
    train_labels.reserve(Constants::CIFAR_TRAIN_SIZE);
    test_labels.reserve(Constants::CIFAR_TEST_SIZE);
}

CIFAR10Dataset::~CIFAR10Dataset() {
    // Vectors clean up automatically
}

bool CIFAR10Dataset::load_batch_file(const std::string& filename,
                                      std::vector<float>& images,
                                      std::vector<int>& labels,
                                      int start_idx) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    // CIFAR-10 binary format:
    // Each record: 1 byte label + 3072 bytes image (32x32x3 in row-major, channel-first)
    // Each batch file contains 10000 images
    
    const int IMAGES_PER_BATCH = 10000;
    const int RECORD_SIZE = 1 + Constants::CIFAR_IMG_PIXELS;
    
    std::vector<unsigned char> buffer(RECORD_SIZE);
    
    for (int i = 0; i < IMAGES_PER_BATCH; ++i) {
        size_t read = fread(buffer.data(), 1, RECORD_SIZE, fp);
        if (read != RECORD_SIZE) {
            std::cerr << "Error: Incomplete read from " << filename << std::endl;
            fclose(fp);
            return false;
        }
        
        // First byte is label
        labels.push_back(static_cast<int>(buffer[0]));
        
        // Remaining bytes are image pixels (already in CHW format)
        // Normalize from [0, 255] to [0, 1]
        for (int j = 0; j < Constants::CIFAR_IMG_PIXELS; ++j) {
            images.push_back(static_cast<float>(buffer[1 + j]) / 255.0f);
        }
    }
    
    fclose(fp);
    return true;
}

bool CIFAR10Dataset::load(const std::string& path) {
    data_path = path;
    
    std::cout << "Loading CIFAR-10 dataset from " << path << std::endl;
    
    // Clear existing data
    train_images.clear();
    train_labels.clear();
    test_images.clear();
    test_labels.clear();
    
    // Load training batches (data_batch_1.bin to data_batch_5.bin)
    for (int batch = 1; batch <= 5; ++batch) {
        std::string filename = path + "/data_batch_" + std::to_string(batch) + ".bin";
        std::cout << "  Loading " << filename << "..." << std::endl;
        
        if (!load_batch_file(filename, train_images, train_labels, (batch - 1) * 10000)) {
            std::cerr << "Failed to load training batch " << batch << std::endl;
            return false;
        }
    }
    
    // Load test batch (test_batch.bin)
    std::string test_filename = path + "/test_batch.bin";
    std::cout << "  Loading " << test_filename << "..." << std::endl;
    
    if (!load_batch_file(test_filename, test_images, test_labels, 0)) {
        std::cerr << "Failed to load test batch" << std::endl;
        return false;
    }
    
    loaded = true;
    
    std::cout << "Dataset loaded successfully!" << std::endl;
    std::cout << "  Training images: " << train_labels.size() << std::endl;
    std::cout << "  Test images: " << test_labels.size() << std::endl;
    
    return true;
}

bool CIFAR10Dataset::download(const std::string& path) {
    std::cout << "Attempting to download CIFAR-10 dataset..." << std::endl;
    
    // Create directory
    std::string mkdir_cmd = "mkdir -p " + path;
    system(mkdir_cmd.c_str());
    
    // Download and extract
    std::string download_cmd = "cd " + path + " && "
        "curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz && "
        "tar -xzf cifar-10-binary.tar.gz && "
        "mv cifar-10-batches-bin/* . && "
        "rmdir cifar-10-batches-bin";
    
    int result = system(download_cmd.c_str());
    if (result != 0) {
        std::cerr << "Error downloading dataset" << std::endl;
        return false;
    }
    
    return load(path);
}

void CIFAR10Dataset::get_image(float* dst, int index, bool is_train) const {
    const float* src = is_train ? train_images.data() : test_images.data();
    int max_idx = is_train ? Constants::CIFAR_TRAIN_SIZE : Constants::CIFAR_TEST_SIZE;
    
    if (index < 0 || index >= max_idx) {
        std::cerr << "Error: Image index out of range" << std::endl;
        return;
    }
    
    memcpy(dst, src + index * Constants::CIFAR_IMG_PIXELS, 
           Constants::CIFAR_IMG_PIXELS * sizeof(float));
}

void CIFAR10Dataset::get_batch(float* dst, const std::vector<int>& indices, bool is_train) const {
    const float* src = is_train ? train_images.data() : test_images.data();
    int max_idx = is_train ? Constants::CIFAR_TRAIN_SIZE : Constants::CIFAR_TEST_SIZE;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (idx < 0 || idx >= max_idx) {
            std::cerr << "Error: Batch index " << idx << " out of range" << std::endl;
            continue;
        }
        memcpy(dst + i * Constants::CIFAR_IMG_PIXELS,
               src + idx * Constants::CIFAR_IMG_PIXELS,
               Constants::CIFAR_IMG_PIXELS * sizeof(float));
    }
}

void CIFAR10Dataset::get_batch_labels(int* dst, const std::vector<int>& indices, bool is_train) const {
    const int* src = is_train ? train_labels.data() : test_labels.data();
    int max_idx = is_train ? Constants::CIFAR_TRAIN_SIZE : Constants::CIFAR_TEST_SIZE;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (idx < 0 || idx >= max_idx) {
            std::cerr << "Error: Label index " << idx << " out of range" << std::endl;
            continue;
        }
        dst[i] = src[idx];
    }
}

const char* CIFAR10Dataset::get_class_name(int label) {
    if (label < 0 || label >= Constants::CIFAR_CLASSES) {
        return "unknown";
    }
    return CIFAR10_CLASSES[label];
}

// ============================================================================
// BatchGenerator Implementation
// ============================================================================

BatchGenerator::BatchGenerator(const CIFAR10Dataset& ds, int batch_sz, bool train, unsigned int seed)
    : dataset(ds), batch_size(batch_sz), current_idx(0), is_train(train), rng(seed) {
    
    int total = is_train ? dataset.get_train_size() : dataset.get_test_size();
    indices.resize(total);
    for (int i = 0; i < total; ++i) {
        indices[i] = i;
    }
}

void BatchGenerator::reset(bool shuffle) {
    current_idx = 0;
    if (shuffle) {
        rng.shuffle(indices);
    }
}

bool BatchGenerator::has_next() const {
    int total = is_train ? dataset.get_train_size() : dataset.get_test_size();
    return current_idx < total;
}

int BatchGenerator::next_batch(float* images, int* labels) {
    int total = is_train ? dataset.get_train_size() : dataset.get_test_size();
    int remaining = total - current_idx;
    int actual_batch_size = std::min(batch_size, remaining);
    
    if (actual_batch_size <= 0) {
        return 0;
    }
    
    // Get batch indices
    std::vector<int> batch_indices(indices.begin() + current_idx,
                                    indices.begin() + current_idx + actual_batch_size);
    
    // Get batch images
    dataset.get_batch(images, batch_indices, is_train);
    
    // Get batch labels if requested
    if (labels != nullptr) {
        dataset.get_batch_labels(labels, batch_indices, is_train);
    }
    
    current_idx += actual_batch_size;
    return actual_batch_size;
}

void BatchGenerator::get_current_indices(std::vector<int>& batch_indices) const {
    int total = is_train ? dataset.get_train_size() : dataset.get_test_size();
    int start = std::max(0, current_idx - batch_size);
    int end = std::min(current_idx, total);
    
    batch_indices.assign(indices.begin() + start, indices.begin() + end);
}

int BatchGenerator::num_batches() const {
    int total = is_train ? dataset.get_train_size() : dataset.get_test_size();
    return (total + batch_size - 1) / batch_size;
}


