# quantoniumos Cryptographic Implementation

> **⚠️ ACADEMIC & SECURITY WARNING**  
> This documentation describes **experimental, novel cryptographic constructions** (RFT-SIS, Enhanced RFT Cipher). These algorithms are **NOT** standard NIST-approved primitives (like AES or SHA-3) and have **NOT** undergone external cryptanalysis or peer review.  
> 
> *   **RFT-SIS Hash**: A novel, non-standard construction unique to this project. It is **NOT** a standard Short Integer Solution (SIS) hash.
> *   **Security Status**: Research Prototype. **DO NOT** use for protecting sensitive production data, real-world financial assets, or high-security systems.
> *   **Verification**: Security claims are based on internal empirical testing only, not on formal reduction proofs or standard compliance.

## Overview

The quantoniumos cryptographic system implements a custom 48-round Feistel cipher with authenticated encryption. The system combines standard cryptographic primitives (for plumbing) with novel RFT-inspired components (for core logic).

---

## Implementation Architecture

### 🏗️ **Cryptographic System Stack**

```
┌─────────────────────────────────────────┐
│           Application Layer             │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │   Q-Vault   │  │   Crypto Tools  │   │
│  │  Encryption │  │   Interface     │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│        Cryptographic API Layer          │
│  ┌─────────────────────────────────────┐ │
│  │    Enhanced RFT Crypto v2           │ │
│  │  encrypt_aead() / decrypt_aead()    │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│          Core Implementation            │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ 48-Round     │  │ HMAC             │ │
│  │ Feistel      │  │ Authentication   │ │
│  │ Cipher       │  │                  │ │
│  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│         Underlying Primitives           │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ AES F-func   │  │ RFT-derived      │ │
│  │ (Standard)   │  │ Key Schedules    │ │
│  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────┘
```

---

## Core Implementation

### 🔐 **Enhanced RFT Crypto v2** (`algorithms/rft/crypto/enhanced_cipher.py`)

**Implementation Features**:
- 48-round Feistel cipher structure
- Authenticated encryption with HMAC verification
- RFT-inspired mixing and key-derived parameters
- AES S-box and MixColumns-style diffusion in the round function
- Domain separation for different use cases

#### **API Functions**

```python
def encrypt_aead(self, plaintext: bytes, associated_data: bytes = b"") -> bytes:
    """
    AEAD-style authenticated encryption
    
    Process:
    1. Generate random salt
    2. Derive per-message keys via HKDF
    3. Apply 48-round Feistel encryption
    4. Generate HMAC authentication tag
    
    Returns: version || salt || ciphertext || mac
    """
```

```python  
def decrypt_aead(self, encrypted_data: bytes, associated_data: bytes = b"") -> bytes:
    """
    AEAD-style authenticated decryption with verification
    
    Process:
    1. Verify HMAC authentication tag
    2. Reconstruct per-message keys from salt
    3. Apply 48-round Feistel decryption
    4. Return plaintext or raise AuthenticationError
    """
```

#### **Core Implementation Components**

##### **48-Round Feistel Structure**
```python
def _feistel_encrypt(self, block: bytes, round_keys: list) -> bytes:
    """
    Feistel network implementation:
    - Block size: 128 bits (16 bytes) 
    - Left/Right halves: 64 bits each
    - Round function: Based on AES components
    
    Standard Feistel structure:
    For round i = 0 to 47:
        L_{i+1} = R_i
        R_{i+1} = L_i ⊕ F(R_i, K_i)
    """
```

##### **F-Function Components**

**AES-Based Round Function:**
```python
def _f_function(self, right_half: bytes, round_key: bytes) -> bytes:
    """
    Round function using standard AES components:
    - S-box substitution for nonlinearity
    - MixColumns-style diffusion
    - Key addition
    """
```

**Key Schedule Generation:**
```python
def _generate_round_keys(self, master_key: bytes, nonce: bytes) -> list:
    """
    Generate 48 round keys using:
    - HKDF key derivation
    - Key-derived phase/amplitude parameters
    - Domain separation
    """
```
##### **Authentication and Verification**

**HMAC Authentication:**
```python
def _generate_authentication_tag(self, ciphertext: bytes, associated_data: bytes, nonce: bytes) -> bytes:
    """
    Generate HMAC authentication tag for AEAD mode
    Uses SHA-256 as underlying hash function
    """
```

**Domain Separation:**
```python
def _domain_separate(self, context: str, data: bytes) -> bytes:
    """
    Provide cryptographic domain separation
    Prevents key reuse across different contexts
    """
```

---

## Performance Characteristics

### 📊 **Measured Performance**

**Current Implementation Results:**
- **Throughput**: 24.0 blocks/sec (128-bit blocks)
- **Latency**: Suitable for interactive applications
- **Memory Usage**: Linear with data size
- **CPU Usage**: Single-threaded Python implementation

**Statistical Validation:**
- **Avalanche Effect**: 50.3% (near-ideal randomness)
- **Differential Uniformity**: Basic validation completed
- **Sample Size**: 1,000 trials (basic level)

---

## Security Properties

### 🔒 **Implemented Security Features**

**Structural Security:**
- 48-round Feistel structure provides high mixing depth
- AES S-box provides nonlinearity
- HMAC provides authentication and integrity
- Domain separation mitigates key reuse

**Implementation Details:**
- Random salt generation for each encryption (via `secrets`)
- Key derivation using standard HKDF
- Authenticated encryption (encrypt-then-MAC)

### ⚠️ **Security Limitations**

**Current Verification Status:**
- **No external audit**: This system has not been reviewed by cryptographers.
- **Novel primitive**: The "RFT-SIS Hash" interaction is unstudied and likely vulnerable.
- **Python-based**: Not constant-time, vulnerable to timing attacks.
- **Non-Standard**: Does not comply with any FIPS or ISO standard.

**Areas for Research (Not Implementation):**
- Proving any security bounds (currently zero proofs exist)
- Converting to constant-time C/Rust
- Removing `numpy` dependencies from all crypto paths

---

## Usage Examples

### 🔧 **Basic Encryption**

```python
from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2

# Initialize with 256-bit key
crypto = EnhancedRFTCryptoV2(b'32-byte-master-key-exactly-256bit')

# Encrypt data
plaintext = b"Message to encrypt"
result = crypto.encrypt_aead(plaintext)

# Result contains: version || salt || ciphertext || mac
```

### 🔓 **Decryption with Verification**

```python
# Decrypt and verify
try:
    decrypted = crypto.decrypt_aead(result)
    print(f"Decrypted: {decrypted}")
except AuthenticationError:
    print("Authentication failed - data may be tampered")
```

---

## Implementation Quality

### ✅ **What Works**

1. **Core Functionality**: 48-round Feistel encryption/decryption
2. **Authentication**: HMAC-based integrity verification  
3. **Key Management**: HKDF-based key derivation
4. **Integration**: Works with quantoniumos applications

### 📋 **Future Enhancements**

1. **Extended Validation**: Scale statistical testing to formal standards
2. **Performance**: C implementation and SIMD optimization
3. **Security Analysis**: Formal cryptographic evaluation
4. **Standards**: Compliance testing and certification

---

## Technical Notes

### � **Implementation Details**

**File Location**: `algorithms/rft/crypto/enhanced_cipher.py` (shim at `algorithms/rft/core/enhanced_rft_crypto_v2.py`)
**Dependencies**: Standard Python cryptographic libraries
**Integration**: Used by Q-Vault and other secure applications
**Testing**: Basic unit tests and statistical validation

**RFT Integration**: The system incorporates RFT-inspired components for:
- Key schedule generation
- Entropy injection
- Phase modulation
- Domain separation

This provides mathematical novelty while maintaining cryptographic soundness through proven primitives.
