

int main() {
    HashData hash;
    hash.allocate(true); // inside allocate, initializeHashParams is called. and the c_hashParams is initialized using config file.

    HashData* d_hashdata; // used to get hash data's GPU memory address.

    int count = 0; // on CPU, initialized to 0 for there are no occupied blocks at the beginning.
    int* d_count; // used to get count's GPU memory address for debug insert and query. Not used for now. TODO: need to fill d_count by calling API.

    // allocate memory for d_hashdata and d_count.
    checkCudaError(cudaMalloc(&d_hashdata, sizeof(HashData)));
    checkCudaError(cudaMalloc(&d_count, sizeof(int)));

    // copy hash data and count to GPU.
    checkCudaError(cudaMemcpy(d_hashdata, &hash, sizeof(HashData), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice));

    // launch test kernel.
    dim3 blockSize(1, 1);
    dim3 gridSize(1000);
    test<<<gridSize, blockSize>>>(d_hashdata, d_count);
    checkCudaError(cudaDeviceSynchronize());

    // free memory.
    hash.free();

    // copy count back to CPU for debug and result check.
    checkCudaError(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "count: " << count << std::endl;

    // free GPU memory.
    checkCudaError(cudaFree(d_hashdata));
    checkCudaError(cudaFree(d_count));

    return 0;
}