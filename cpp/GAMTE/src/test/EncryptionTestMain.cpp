#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <fstream>
#include <CL/cl.hpp> // OpenCL header for GPU offloading

// Custom Encryptor Class with Multithreading and GPU Offloading
class CustomEncryptor {
public:
    CustomEncryptor(const std::vector<uint8_t>& key, size_t dataSize) : key(key), numRounds(40) {
        if (key.size() != 512) {
            throw std::invalid_argument("Key must be 4096 bits (512 bytes) long.");
        }
        initializeSBoxFromKey();
        initializePBox(dataSize);  // Initialize pBox based on data size
        expandKey();
        initializeOpenCL();
    }

    // Multithreaded encryption function
    std::vector<uint8_t> parallelEncrypt(const std::vector<uint8_t>& plaintext, int numThreads) {
        size_t dataSize = plaintext.size();
        std::vector<uint8_t> ciphertext(dataSize);
        size_t chunkSize = dataSize / numThreads;

        std::vector<std::thread> threads;
        for (int i = 0; i < numThreads; ++i) {
            size_t start = i * chunkSize;
            size_t end = (i == numThreads - 1) ? dataSize : start + chunkSize;
            std::vector<uint8_t> chunk(plaintext.begin() + start, plaintext.begin() + end);

            threads.emplace_back([&](std::vector<uint8_t>& part, size_t offset) {
                std::vector<uint8_t> encryptedChunk = encrypt(part);
                std::copy(encryptedChunk.begin(), encryptedChunk.end(), ciphertext.begin() + offset);
                }, std::ref(chunk), start);
        }

        for (auto& t : threads) {
            t.join();
        }

        return ciphertext;
    }

    // Multithreaded decryption function
    std::vector<uint8_t> parallelDecrypt(const std::vector<uint8_t>& ciphertext, int numThreads) {
        size_t dataSize = ciphertext.size();
        std::vector<uint8_t> decryptedText(dataSize);
        size_t chunkSize = dataSize / numThreads;

        std::vector<std::thread> threads;
        for (int i = 0; i < numThreads; ++i) {
            size_t start = i * chunkSize;
            size_t end = (i == numThreads - 1) ? dataSize : start + chunkSize;
            std::vector<uint8_t> chunk(ciphertext.begin() + start, ciphertext.begin() + end);

            threads.emplace_back([&](std::vector<uint8_t>& part, size_t offset) {
                std::vector<uint8_t> decryptedChunk = decrypt(part);
                std::copy(decryptedChunk.begin(), decryptedChunk.end(), decryptedText.begin() + offset);
                }, std::ref(chunk), start);
        }

        for (auto& t : threads) {
            t.join();
        }

        return decryptedText;
    }

private:
    std::vector<uint8_t> key;
    std::vector<uint8_t> sBox;
    std::vector<uint8_t> pBox;
    std::vector<std::vector<uint8_t>> roundKeys;
    int numRounds;

    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;

    void initializeSBoxFromKey() {
        sBox.resize(256);
        std::iota(sBox.begin(), sBox.end(), 0);
        std::seed_seq seed(key.begin(), key.end());
        std::mt19937 rng(seed);
        std::shuffle(sBox.begin(), sBox.end(), rng);
    }

    void initializePBox(size_t dataSize) {
        pBox.resize(dataSize);
        std::iota(pBox.begin(), pBox.end(), 0);
        std::shuffle(pBox.begin(), pBox.end(), std::mt19937{ std::random_device{}() });
    }

    void expandKey() {
        roundKeys.resize(numRounds + 1, std::vector<uint8_t>(key.size()));
        for (int i = 0; i < numRounds + 1; ++i) {
            roundKeys[i] = key;
            std::rotate(roundKeys[i].begin(), roundKeys[i].begin() + (i % roundKeys[i].size()), roundKeys[i].end());
        }
    }

    // Initialize OpenCL for GPU Offloading
    void initializeOpenCL() {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms.front();

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Device device = devices.front();

        context = cl::Context(device);
        queue = cl::CommandQueue(context, device);

        // Define the OpenCL kernel for XOR operation
        const char* kernelCode = R"(
            __kernel void xorOperation(__global const uchar* data, __global const uchar* roundKey, __global uchar* result, int size) {
                int id = get_global_id(0);
                if (id < size) {
                    result[id] = data[id] ^ roundKey[id % size];
                }
            }
        )";

        cl::Program::Sources sources;
        sources.push_back({ kernelCode, strlen(kernelCode) });
        program = cl::Program(context, sources);
        program.build({ device });
    }

    // Perform XOR using GPU
    void gpuXOR(std::vector<uint8_t>& data, const std::vector<uint8_t>& roundKey) {
        cl::Buffer bufferData(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data.size(), data.data());
        cl::Buffer bufferKey(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, roundKey.size(), const_cast<uint8_t*>(roundKey.data()));
        cl::Buffer bufferResult(context, CL_MEM_WRITE_ONLY, data.size());

        cl::Kernel kernel(program, "xorOperation");
        kernel.setArg(0, bufferData);
        kernel.setArg(1, bufferKey);
        kernel.setArg(2, bufferResult);
        kernel.setArg(3, static_cast<int>(data.size()));

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(data.size()), cl::NullRange);
        queue.enqueueReadBuffer(bufferResult, CL_TRUE, 0, data.size(), data.data());
    }

    std::vector<uint8_t> encrypt(const std::vector<uint8_t>& plaintext) {
        std::vector<uint8_t> ciphertext = plaintext;
        for (int i = 0; i < numRounds; ++i) {
            addRoundKey(ciphertext, roundKeys[i]);
            substitution(ciphertext);
            permutation(ciphertext);
            gpuXOR(ciphertext, roundKeys[i]);  // Perform XOR using GPU offloading
        }
        addRoundKey(ciphertext, roundKeys[numRounds]);
        return ciphertext;
    }

    std::vector<uint8_t> decrypt(const std::vector<uint8_t>& ciphertext) {
        std::vector<uint8_t> plaintext = ciphertext;
        addRoundKey(plaintext, roundKeys[numRounds]);
        for (int i = numRounds - 1; i >= 0; --i) {
            gpuXOR(plaintext, roundKeys[i]);  // Perform XOR using GPU offloading
            inversePermutation(plaintext);
            inverseSubstitution(plaintext);
            addRoundKey(plaintext, roundKeys[i]);
        }
        return plaintext;
    }

    void addRoundKey(std::vector<uint8_t>& data, const std::vector<uint8_t>& roundKey) {
        gpuXOR(data, roundKey);
    }

    void substitution(std::vector<uint8_t>& data) {
        for (auto& byte : data) {
            byte = sBox[byte];
        }
    }

    void permutation(std::vector<uint8_t>& data) {
        std::vector<uint8_t> temp(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            temp[i] = data[pBox[i] % data.size()];
        }
        data = temp;
    }

    void inverseSubstitution(std::vector<uint8_t>& data) {
        for (auto& byte : data) {
            byte = std::find(sBox.begin(), sBox.end(), byte) - sBox.begin();
        }
    }

    void inversePermutation(std::vector<uint8_t>& data) {
        std::vector<uint8_t> temp(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            temp[pBox[i] % data.size()] = data[i];
        }
        data = temp;
    }
};

// Utility function to print data in hexadecimal
void printHex(const std::vector<uint8_t>& data) {
    for (auto byte : data) {
        std::cout << std::hex << (int)byte << " ";
    }
    std::cout << std::dec << std::endl;
}

// Performance reporting
void reportPerformance(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end, const std::string& operation) {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << operation << " completed in " << duration << " milliseconds." << std::endl;
}

int main() {
    try {
        // Step 1: Generate a random key
        std::vector<uint8_t> key(512);  // 4096 bits
        std::generate(key.begin(), key.end(), std::rand);

        // Step 2: Prepare a plaintext for encryption
        std::string message = "Multithreaded and GPU-offloaded encryption test message.";
        std::vector<uint8_t> plaintext(message.begin(), message.end());

        // Step 3: Initialize the encryption system
        CustomEncryptor encryptor(key, plaintext.size());

        std::cout << "Original plaintext: " << message << std::endl;
        std::cout << "Plaintext (hex): ";
        printHex(plaintext);

        // Step 4: Encrypt the plaintext using multithreading
        int numThreads = std::thread::hardware_concurrency();  // Use the number of available CPU cores
        auto startEncrypt = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> ciphertext = encryptor.parallelEncrypt(plaintext, numThreads);
        auto endEncrypt = std::chrono::high_resolution_clock::now();
        reportPerformance(startEncrypt, endEncrypt, "Encryption");

        std::cout << "Ciphertext (hex): ";
        printHex(ciphertext);

        // Step 5: Decrypt the ciphertext using multithreading
        auto startDecrypt = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> decryptedText = encryptor.parallelDecrypt(ciphertext, numThreads);
        auto endDecrypt = std::chrono::high_resolution_clock::now();
        reportPerformance(startDecrypt, endDecrypt, "Decryption");

        std::string decryptedMessage(decryptedText.begin(), decryptedText.end());

        std::cout << "Decrypted text: " << decryptedMessage << std::endl;
        std::cout << "Decrypted text (hex): ";
        printHex(decryptedText);

        // Step 6: Verify that decryption matches the original plaintext
        if (plaintext == decryptedText) {
            std::cout << "Encryption/Decryption test PASSED." << std::endl;
        }
        else {
            std::cout << "Encryption/Decryption test FAILED." << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}