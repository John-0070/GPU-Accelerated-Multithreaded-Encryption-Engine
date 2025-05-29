# CustomEncryptor

## Description

Custom Encryption is a high-performance, experimental POC encryption engine written in C++ that utilizes both multithreading and GPU offloading (via OpenCL) to perform symmetric encryption using a custom round-based transformation routine. This proof-of-concept (PoC) explores efficient, hardware-accelerated cryptographic processing with a strong emphasis on parallelization and performance tuning.

---

## Features

- **4096-bit (512-byte) symmetric key support**
- **40-round block encryption/decryption pipeline**
- **Multithreaded CPU execution using `std::thread`**
- **GPU-accelerated XOR operations via OpenCL**
- **Custom S-box (substitution box) derived from key material**
- **Dynamic P-box (permutation box) based on data length**
- **Key expansion per round via rotation**
- **Hex dump utilities and performance reporting**

---

## Technical Workflow

### Encryption Routine

Each round of encryption performs:

1. `AddRoundKey` – XOR using the round key (executed on GPU)
2. `Substitution` – Byte substitution using a shuffled S-box
3. `Permutation` – Reordering bytes using a P-box
4. `GPU XOR` – Offloaded XOR using OpenCL

### Decryption Routine

Decryption applies the inverse of the above in reverse order:

1. `GPU XOR` (reverse of last round key)
2. `Inverse Permutation`
3. `Inverse Substitution`
4. `AddRoundKey` again

---

## GPU Integration

A simple OpenCL kernel performs the XOR operation:

``c
__kernel void xorOperation(__global const uchar* data, __global const uchar* roundKey, __global uchar* result, int size) {
    int id = get_global_id(0);
    if (id < size) {
        result[id] = data[id] ^ roundKey[id % size];
    }
}
This is compiled and executed at runtime using the OpenCL C++ bindings (cl.hpp), targeting the default available GPU device.

# Build & Run Requirements
C++17 or newer

OpenCL-compatible GPU

OpenCL SDK and runtime installed

C++ compiler supporting cl.hpp (e.g., MSVC, GCC, Clang)

# Dependencies
CL/cl.hpp

Standard C++ STL: <thread>, <vector>, <random>, <chrono>, <numeric>, <stdexcept>, <algorithm>

# Example Usage (Main)
std::vector<uint8_t> key(512);  // 4096-bit key
std::generate(key.begin(), key.end(), std::rand);

std::string message = "Multithreaded and GPU-offloaded encryption test message.";
std::vector<uint8_t> plaintext(message.begin(), message.end());

CustomEncryptor encryptor(key, plaintext.size());
auto ciphertext = encryptor.parallelEncrypt(plaintext, std::thread::hardware_concurrency());
auto decrypted = encryptor.parallelDecrypt(ciphertext, std::thread::hardware_concurrency());

# Output Sample
- Original plaintext: 

Multithreaded and GPU-offloaded encryption test message.
Plaintext (hex): 4d 75 6c 74 69 ...
Ciphertext (hex): a1 c4 3f ...
Decrypted text: Multithreaded and GPU-offloaded encryption test message.
Encryption completed in X ms.
Decryption completed in Y ms.
Encryption/Decryption test PASSED.

# Limitations
- No authenticated encryption or integrity validation (e.g., MAC, HMAC, AEAD)

- No support for asymmetric encryption or key exchange

- No padding scheme included (assumes plaintext size alignment)

- XOR offload is the only GPU-accelerated primitive

- Only supports static configuration (fixed rounds, key size)

- No fallback if GPU is not available

- No stream cipher support (CBC/CTR/CFB modes not implemented)

- S-box and P-box logic is custom, not cryptographically vetted

# Potential Improvements
- Add GCM/CTR mode support and secure padding

- Replace XOR kernel with real S-box kernel logic

- Implement full OpenCL fallback to CPU-only pipeline

- Parameterize number of rounds and key sizes

- Use memory pools to avoid frequent allocations

- Add authenticated encryption layer (e.g., Poly1305 or HMAC)

- Add dynamic CLI or config support for tuning execution

# Use Cases
- Demonstration of multithreaded and GPU-parallelizable encryption

- Performance benchmarking for hybrid cryptographic operations

- Educational example for combining C++ concurrency and OpenCL

- Research on round-based custom cipher design

- Red team tool prototype for obfuscated data transport
