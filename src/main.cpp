#include "utils.h"
#include "data_loader.h"
#include "autoencoder.h"
#include <cstring>
#include <cstdlib>

// Forward declarations for GPU functions (defined in autoencoder_gpu.cu)
#ifndef CPU_ONLY
void train_gpu(CIFAR10Dataset& dataset, const TrainingConfig& config, TrainingStats& stats);
void train_gpu_optimized(CIFAR10Dataset& dataset, const TrainingConfig& config, TrainingStats& stats);
void extract_features_gpu(CIFAR10Dataset& dataset, const std::string& model_path,
                          float* train_features, float* test_features);
#endif

// ============================================================================
// CPU Training Function
// ============================================================================

void train_cpu(CIFAR10Dataset& dataset, const TrainingConfig& config, TrainingStats& stats) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "CPU Training" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Learning rate: " << config.learning_rate << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Initialize autoencoder
    CPUAutoencoder autoencoder;
    autoencoder.initialize(config.batch_size, config.seed);
    
    // Create batch generator
    BatchGenerator batch_gen(dataset, config.batch_size, true, config.seed);
    
    // Allocate batch buffer
    float* batch_images = new float[config.batch_size * Constants::CIFAR_IMG_PIXELS];
    
    Timer total_timer("Total Training");
    
    // Training loop
    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        Timer epoch_timer("Epoch");
        batch_gen.reset(true);  // Shuffle at each epoch
        
        float epoch_loss = 0.0f;
        int num_batches = 0;
        int batch_idx = 0;
        
        while (batch_gen.has_next()) {
            int actual_batch_size = batch_gen.next_batch(batch_images);
            
            if (actual_batch_size < config.batch_size) {
                // Skip incomplete batches for simplicity
                continue;
            }
            
            // Forward pass
            float loss = autoencoder.forward(batch_images, nullptr);
            
            // Backward pass
            autoencoder.backward(batch_images);
            
            // Update weights
            autoencoder.update_weights(config.learning_rate);
            
            epoch_loss += loss;
            num_batches++;
            batch_idx++;
            
            // Print progress
            if (batch_idx % config.print_every == 0) {
                print_progress(batch_idx, batch_gen.num_batches(), epoch_loss / num_batches);
            }
        }
        
        epoch_loss /= num_batches;
        double epoch_time = epoch_timer.elapsed();
        
        stats.epoch_losses.push_back(epoch_loss);
        stats.epoch_times.push_back(epoch_time);
        
        std::cout << std::endl;
        std::cout << "Epoch " << epoch + 1 << "/" << config.epochs 
                  << " - Loss: " << epoch_loss 
                  << " - Time: " << epoch_time << "s" << std::endl;
    }
    
    stats.total_time = total_timer.elapsed();
    stats.final_loss = stats.epoch_losses.back();
    
    // Save model
    autoencoder.save_weights(config.save_path);
    
    // Clean up
    delete[] batch_images;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Training Complete!" << std::endl;
    std::cout << "Total time: " << stats.total_time << "s" << std::endl;
    std::cout << "Final loss: " << stats.final_loss << std::endl;
    std::cout << "Model saved to: " << config.save_path << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ============================================================================
// Feature Extraction (CPU)
// ============================================================================

void extract_features_cpu(CIFAR10Dataset& dataset, const std::string& model_path,
                          float* train_features, float* test_features) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "CPU Feature Extraction" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    Timer timer("Feature Extraction");
    
    const int batch_size = 100;  // Process 100 images at a time
    
    // Initialize autoencoder and load weights
    CPUAutoencoder autoencoder;
    autoencoder.initialize(batch_size);
    
    if (!autoencoder.load_weights(model_path)) {
        std::cerr << "Error: Failed to load model weights" << std::endl;
        return;
    }
    
    float* batch_images = new float[batch_size * Constants::CIFAR_IMG_PIXELS];
    float* batch_features = new float[batch_size * Constants::FEATURE_DIM];
    
    // Extract training features
    std::cout << "Extracting training features..." << std::endl;
    for (int i = 0; i < Constants::CIFAR_TRAIN_SIZE; i += batch_size) {
        int actual_batch = std::min(batch_size, Constants::CIFAR_TRAIN_SIZE - i);
        
        // Get batch of images
        std::vector<int> indices(actual_batch);
        for (int j = 0; j < actual_batch; ++j) {
            indices[j] = i + j;
        }
        dataset.get_batch(batch_images, indices, true);
        
        // Extract features
        autoencoder.encode(batch_images, batch_features);
        
        // Copy to output buffer
        memcpy(train_features + i * Constants::FEATURE_DIM,
               batch_features, actual_batch * Constants::FEATURE_DIM * sizeof(float));
        
        if ((i / batch_size) % 50 == 0) {
            print_progress(i, Constants::CIFAR_TRAIN_SIZE);
        }
    }
    std::cout << std::endl;
    
    // Extract test features
    std::cout << "Extracting test features..." << std::endl;
    for (int i = 0; i < Constants::CIFAR_TEST_SIZE; i += batch_size) {
        int actual_batch = std::min(batch_size, Constants::CIFAR_TEST_SIZE - i);
        
        std::vector<int> indices(actual_batch);
        for (int j = 0; j < actual_batch; ++j) {
            indices[j] = i + j;
        }
        dataset.get_batch(batch_images, indices, false);
        
        autoencoder.encode(batch_images, batch_features);
        
        memcpy(test_features + i * Constants::FEATURE_DIM,
               batch_features, actual_batch * Constants::FEATURE_DIM * sizeof(float));
        
        if ((i / batch_size) % 10 == 0) {
            print_progress(i, Constants::CIFAR_TEST_SIZE);
        }
    }
    std::cout << std::endl;
    
    delete[] batch_images;
    delete[] batch_features;
    
    double elapsed = timer.elapsed();
    std::cout << "Feature extraction completed in " << elapsed << "s" << std::endl;
    std::cout << "Train features shape: (" << Constants::CIFAR_TRAIN_SIZE << ", " 
              << Constants::FEATURE_DIM << ")" << std::endl;
    std::cout << "Test features shape: (" << Constants::CIFAR_TEST_SIZE << ", " 
              << Constants::FEATURE_DIM << ")" << std::endl;
}

// ============================================================================
// Command Line Argument Parsing
// ============================================================================

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --train              Train the autoencoder" << std::endl;
    std::cout << "  --test               Run tests" << std::endl;
    std::cout << "  --extract-features   Extract features using trained encoder" << std::endl;
    std::cout << "  --train-svm          Train SVM on extracted features" << std::endl;
    std::cout << "  --evaluate           Evaluate on test set" << std::endl;
    std::cout << "  --cpu                Use CPU only" << std::endl;
    std::cout << "  --optimized          Use optimized GPU kernels" << std::endl;
    std::cout << "  --epochs N           Number of training epochs (default: 20)" << std::endl;
    std::cout << "  --batch-size N       Batch size (default: 64)" << std::endl;
    std::cout << "  --lr RATE            Learning rate (default: 0.001)" << std::endl;
    std::cout << "  --data PATH          Path to CIFAR-10 data (default: data/)" << std::endl;
    std::cout << "  --model PATH         Path to save/load model (default: models/autoencoder.bin)" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
}

struct CommandLineArgs {
    bool train = false;
    bool test = false;
    bool extract_features = false;
    bool train_svm = false;
    bool evaluate = false;
    bool cpu_only = false;
    bool optimized = false;
    int epochs = 20;
    int batch_size = 64;
    float learning_rate = 0.001f;
    std::string data_path = "data/";
    std::string model_path = "models/autoencoder.bin";
};

CommandLineArgs parse_args(int argc, char* argv[]) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--train") args.train = true;
        else if (arg == "--test") args.test = true;
        else if (arg == "--extract-features") args.extract_features = true;
        else if (arg == "--train-svm") args.train_svm = true;
        else if (arg == "--evaluate") args.evaluate = true;
        else if (arg == "--cpu") args.cpu_only = true;
        else if (arg == "--optimized") args.optimized = true;
        else if (arg == "--epochs" && i + 1 < argc) args.epochs = std::atoi(argv[++i]);
        else if (arg == "--batch-size" && i + 1 < argc) args.batch_size = std::atoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) args.learning_rate = std::atof(argv[++i]);
        else if (arg == "--data" && i + 1 < argc) args.data_path = argv[++i];
        else if (arg == "--model" && i + 1 < argc) args.model_path = argv[++i];
        else if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
    }
    
    return args;
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "CUDA Autoencoder for CIFAR-10" << std::endl;
    std::cout << "CSC14120 - Parallel Programming" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Parse command line arguments
    CommandLineArgs args = parse_args(argc, argv);
    
    // If no action specified, show usage
    if (!args.train && !args.test && !args.extract_features && 
        !args.train_svm && !args.evaluate) {
        print_usage(argv[0]);
        return 0;
    }
    
    // Load dataset
    CIFAR10Dataset dataset;
    if (!dataset.load(args.data_path)) {
        std::cerr << "Error: Failed to load CIFAR-10 dataset from " << args.data_path << std::endl;
        std::cerr << "Try running 'make download_data' first." << std::endl;
        return 1;
    }
    
    // Configure training
    TrainingConfig config;
    config.batch_size = args.batch_size;
    config.epochs = args.epochs;
    config.learning_rate = args.learning_rate;
    config.use_gpu = !args.cpu_only;
    config.use_optimized = args.optimized;
    config.save_path = args.model_path;
    
    TrainingStats stats;
    
    // Run training
    if (args.train) {
        #ifdef CPU_ONLY
        train_cpu(dataset, config, stats);
        #else
        if (args.cpu_only) {
            train_cpu(dataset, config, stats);
        } else if (args.optimized) {
            train_gpu_optimized(dataset, config, stats);
        } else {
            train_gpu(dataset, config, stats);
        }
        #endif
        
        // Print training summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "Training Summary" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total training time: " << stats.total_time << "s" << std::endl;
        std::cout << "Average epoch time: " << (stats.total_time / args.epochs) << "s" << std::endl;
        std::cout << "Final loss: " << stats.final_loss << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
    
    // Extract features
    if (args.extract_features) {
        float* train_features = new float[Constants::CIFAR_TRAIN_SIZE * Constants::FEATURE_DIM];
        float* test_features = new float[Constants::CIFAR_TEST_SIZE * Constants::FEATURE_DIM];
        
        #ifdef CPU_ONLY
        extract_features_cpu(dataset, args.model_path, train_features, test_features);
        #else
        if (args.cpu_only) {
            extract_features_cpu(dataset, args.model_path, train_features, test_features);
        } else {
            extract_features_gpu(dataset, args.model_path, train_features, test_features);
        }
        #endif
        
        // Save features to file for SVM training
        std::string train_features_path = "models/train_features.bin";
        std::string test_features_path = "models/test_features.bin";
        
        std::ofstream train_file(train_features_path, std::ios::binary);
        train_file.write(reinterpret_cast<char*>(train_features), 
                        Constants::CIFAR_TRAIN_SIZE * Constants::FEATURE_DIM * sizeof(float));
        train_file.close();
        
        std::ofstream test_file(test_features_path, std::ios::binary);
        test_file.write(reinterpret_cast<char*>(test_features), 
                       Constants::CIFAR_TEST_SIZE * Constants::FEATURE_DIM * sizeof(float));
        test_file.close();
        
        std::cout << "Features saved to:" << std::endl;
        std::cout << "  Train: " << train_features_path << std::endl;
        std::cout << "  Test: " << test_features_path << std::endl;
        
        delete[] train_features;
        delete[] test_features;
    }
    
    // Run tests
    if (args.test) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running Tests" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        // Test 1: Data loading
        std::cout << "Test 1: Data loading... ";
        if (dataset.is_loaded()) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
        }
        
        // Test 2: Forward pass (CPU)
        std::cout << "Test 2: CPU forward pass... ";
        CPUAutoencoder cpu_ae;
        cpu_ae.initialize(1);
        float test_image[Constants::CIFAR_IMG_PIXELS];
        float test_output[Constants::CIFAR_IMG_PIXELS];
        dataset.get_image(test_image, 0, true);
        float loss = cpu_ae.forward(test_image, test_output);
        if (loss > 0 && loss < 10) {
            std::cout << "PASSED (loss=" << loss << ")" << std::endl;
        } else {
            std::cout << "FAILED (loss=" << loss << ")" << std::endl;
        }
        
        // Test 3: Backward pass (CPU)
        std::cout << "Test 3: CPU backward pass... ";
        cpu_ae.backward(test_image);
        std::cout << "PASSED" << std::endl;
        
        std::cout << "\n========================================" << std::endl;
    }
    
    return 0;
}

