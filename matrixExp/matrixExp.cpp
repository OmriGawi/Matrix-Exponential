#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <thread>
#include <cmath>

using namespace std;

vector<vector<float>> calcTylor(vector<vector<float>> matrix);
vector<vector<float>> get2DVector(int n, string filename);
void print2DVector(vector<vector<float>> vec);
vector<vector<float>> initUnitMatrix(int n, int m);
vector<vector<float>> sumTwoMatrix(vector<vector<float>> matrix1, vector<vector<float>> matrix2);
vector<vector<float>> multiplyValueToMatrix(vector<vector<float>> matrix, float k);
float* matrixToArr(const vector<vector<float>>& vec);
vector<vector<float>> arrToMatrix(float* arr, int rows, int cols);
void addRowsAndColumnsToMatrix(vector<vector<float>>& matrix, int rowsToAdd, int colsToAdd);

float power_iteration(vector<vector<float>> A);
float eigenvalue(vector<vector<float>> A, vector<float> v, int n);
vector<float> calcAv(vector<vector<float>> A, vector<float> v);
float calcVv(vector<float> v1, vector<float> v2);

void init_c(int M, int N, float* C, int ldc);
void reorder(int K, const float* B, int ldb, float* B_tmp);
void kernel(int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc);
void mult(int M, int K, int N, const float* A, const float* B, float* C, int lda, int ldb, int ldc);
float* mult_thread(int M, const float* A, const float* B, int lda, int ldb, int ldc);

float* Random(size_t n);
long Get_Time();


int main() {
    int t1, t2;
    t1 = Get_Time();
    vector<vector<float>> matrix = get2DVector(1000, "inv_matrix(1000x1000).txt");
    vector<vector<float>> tylor_matrix = calcTylor(matrix);
    t2 = Get_Time();

    cout << t2 - t1 << endl;

    // free memory
        //matrix.~vector<vector<float>>();
        //tylor_matrix.~vector<vector<float>>();

    return 0;
}

vector<vector<float>> calcTylor(vector<vector<float>> matrix) {
    // get the size of matrix
    int row = matrix.size();
    int col = matrix[0].size();
    // create a unit matrix depends on the size of the given matrix
    vector<vector<float>> unitMatrix = initUnitMatrix(row, col);
    // find the C value for Lagrange reminder
    float c = (float)sqrt(power_iteration(matrix));

    float epsilon = 0.0000000001, coefficient, coefficient_reminder, reminder;
    int multiplyedDenom = 1;
    int i = 2;
    int extension = 1304;
    int n = matrix.size();

    vector<vector<float>> tylorMatrix = sumTwoMatrix(unitMatrix, matrix);
    vector<vector<float>> multiplyedMatrix = matrix;
    vector<vector<float>> temp_matrix;

    // change the size of the multiplyed matrix
    addRowsAndColumnsToMatrix(multiplyedMatrix, 1304, 1304);
    float* arr_multiplyedMatrix = matrixToArr(multiplyedMatrix);

    // change the size of the matrix
    addRowsAndColumnsToMatrix(matrix, 1304, 1304);
    float* arr_matrix = matrixToArr(matrix);

    while (true) {
        arr_multiplyedMatrix = mult_thread(2304, arr_multiplyedMatrix, arr_matrix, 2304, 2304, 2304);

        multiplyedDenom = multiplyedDenom * i;
        coefficient = (float)(1.0 / multiplyedDenom);
        multiplyedMatrix = arrToMatrix(arr_multiplyedMatrix, row + extension, col + extension);
        addRowsAndColumnsToMatrix(multiplyedMatrix, -1304, -1304);
        temp_matrix = multiplyValueToMatrix(multiplyedMatrix, coefficient);
        tylorMatrix = sumTwoMatrix(tylorMatrix, temp_matrix);
        // free memory
        temp_matrix.~vector<vector<float>>();
        // calculate reminder
        coefficient_reminder = (float)pow(M_E, c) / multiplyedDenom * (i + 1);
        reminder = coefficient_reminder * (float)pow(c, i + 1);
        i = i + 1;

        if (reminder < epsilon)
            break;
    }
    return tylorMatrix;
}


/// <summary>
/// read 2D vector from fileName
/// </summary>
/// <param name="n">: size of 2D vector</param>
/// <param name="filename">: file name</param>
/// <returns>: 2D vector</returns>
vector<vector<float>> get2DVector(int n, string filename) {
    vector<vector<float>> result(n);
    ifstream input(filename);
    string s;
    for (int i = 0; i < n; i++) {
        getline(input, s);
        istringstream iss(s);

        string num;
        while (std::getline(iss, num, ','))
            result[i].push_back(std::stof(num));
    }
    return result;
}

/// <summary>
/// print 2D vector
/// </summary>
/// <param name="vec">: 2D vector to print</param>
void print2DVector(vector<vector<float>> vec) {
    for (const auto& row : vec) {
        for (const auto& value : row) {
            cout << value << "\t";
        }
        cout << endl;
    }
}

/// <summary>
/// initialize a 2D vector as a unit matrix
/// </summary>
/// <param name="n">: row size</param>
/// <param name="m">: col size</param>
/// <returns> unit matrix</returns>
vector<vector<float>> initUnitMatrix(int n, int m) {
    vector<vector<float>> unitMatrix(n, vector<float>(m, 0));
    for (int i = 0; i < n; i++)
        unitMatrix[i][i] = 1;
    return unitMatrix;
}

/// <summary>
/// sum two matrix
/// </summary>
/// <param name="matrix1">: first matrix</param>
/// <param name="matrix2">: second matrix</param>
/// <returns> the sum of two given matrix</returns>
vector<vector<float>> sumTwoMatrix(vector<vector<float>> matrix1, vector<vector<float>> matrix2) {
    int row = matrix1.size();
    int col = matrix1[0].size();
    vector<vector<float>> new_matrix(row, vector<float>(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            new_matrix[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
    return new_matrix;
}

/// <summary>
/// Multiply all matrix value by a given k value.
/// </summary>
/// <param name="matrix">matrix to be multiplyed</param>
/// <param name="k">value to multiply the matrix</param>
/// <returns>matrix</returns>
vector<vector<float>> multiplyValueToMatrix(vector<vector<float>> matrix, float k) {
    vector<vector<float>> new_matrix(matrix.size(), vector<float>(matrix[0].size(), 0));
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[0].size(); j++) {
            new_matrix[i][j] = matrix[i][j] * k;
        }
    }
    return new_matrix;
}

/// <summary>
/// change matrix to array
/// </summary>
/// <param name="vec"></param>
/// <returns>new array</returns>
float* matrixToArr(const vector<vector<float>>& vec) {
    float* newarr = new float[vec.size() * vec[0].size()];
    float* newarr_ptr_copy = newarr;
    for (int i = 0; i < vec.size(); i++) {
        std::copy(vec[i].begin(), vec[i].end(), newarr_ptr_copy);
        newarr_ptr_copy += vec[i].size(); // don't lose track of newarr
    }
    return newarr;
}

/// <summary>
/// change given array to matrix
/// </summary>
/// <param name="arr"></param>
/// <param name="rows"></param>
/// <param name="cols"></param>
/// <returns></returns>
vector<vector<float>> arrToMatrix(float* arr, int rows, int cols) {
    vector<vector<float>> matrix;
    for (int i = 0; i < rows; i++) {
        vector<float> row;
        for (int j = 0; j < cols; j++) {
            row.push_back(arr[i * cols + j]);
        }
        matrix.push_back(row);
    }
    return matrix;
}

/// <summary>
/// add rows and columns to matrix
/// </summary>
/// <param name="matrix">matrix to add to</param>
/// <param name="rowsToAdd">number of rows to add</param>
/// <param name="colsToAdd">number of columns to add</param>
void addRowsAndColumnsToMatrix(vector<vector<float>>& matrix, int rowsToAdd, int colsToAdd) {
    int old_matrix_size = matrix.size();
    // Add rowsToAdd rows to the matrix
    matrix.resize(matrix.size() + rowsToAdd);

    // Add colsToAdd columns to each row and fill the new values with 0
    for (int i = 0; i < matrix.size(); i++) {
        if (i >= old_matrix_size) {
            matrix[i].resize(old_matrix_size + colsToAdd, 0);
        }
        else {
            matrix[i].resize(matrix[i].size() + colsToAdd, 0);
        }

    }
}


/// <summary>
/// calculates the norm of a given vector
/// </summary>
/// <param name="v">: vector</param>
/// <returns>norm</returns>
float calcVectorNorm(vector<float> v) {
    return (float)sqrt(calcVv(v, v));
}

/// <summary>
/// calculates the eigenvalue (Av = xV) 
/// </summary>
/// <param name="A">: 2D vector</param>
/// <param name="v">: vector</param>
/// <param name="n">: size of A</param>
/// <returns>eigenvalue</returns>
float eigenvalue(vector<vector<float>> A, vector<float> v, int n) {
    vector<float> Av(n);
    int row = A.size();
    int col = A[0].size();
    float sum = 0;
    float ev = 0;

    // calculate A.dot(v)
    Av = calcAv(A, v);

    // calculate v.dot(Av)
    ev = calcVv(v, Av);

    return ev;
}

/// <summary>
/// calculate the biggest eigenvalue of a given 2D vector
/// </summary>
/// <param name="A">: 2D vector</param>
/// <returns>the biggesst eigenvalue</returns>
float power_iteration(vector<vector<float>> A) {
    int n = A.size();
    int row = A.size();
    int col = A[0].size();

    float sum = 0;
    float random_value, ev, ev_new, norm_Av;

    vector<float> v(n), Av(n), v_new(n);

    random_value = 1 / (float)sqrt(col);

    for (int i = 0; i < n; i++) {
        v[i] = random_value;
    }

    ev = eigenvalue(A, v, n);

    while (true) {
        Av = calcAv(A, v);
        norm_Av = calcVectorNorm(Av);

        // calc v_new
        for (int i = 0; i < n; i++) {
            v_new[i] = Av[i] / norm_Av;
        }

        ev_new = eigenvalue(A, v_new, n);
        if ((float)abs(ev - ev_new) < 0.0000000001)
            break;
        
        // ev for the next iteration
        ev = ev_new;
    }

    return ev_new;
}

/// <summary>
/// calculate the multiply of 2 vectors
/// </summary>
/// <param name="v1">: vector1</param>
/// <param name="v2">: vector2</param>
/// <returns> returns the multiply of 2 vectoor</returns>
float calcVv(vector<float> v1, vector<float> v2) {
    int size = v1.size();
    float sum = 0;

    for(int i = 0; i < size; i++) {
        sum = sum + v1[i] * v2[i];
    }
    return sum;
}

/// <summary>
/// calculate the multiply of 2D vector and vector
/// </summary>
/// <param name="A">: 2D vector</param>
/// <param name="v">: vector</param>
/// <returns>returns the multiply of 2D vector and vector</returns>
vector<float> calcAv(vector<vector<float>> A, vector<float> v) {
    int row = A.size();
    int col = A[0].size();
    float sum = 0;
    vector<float> Av(A.size());

    // calculate A.dot(v)
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            sum = sum + A[i][j] * v[j];
        }
        Av[i] = sum;
        sum = 0;
    }
    return Av;
}


// Multithreaded product of matrices A (M x K) and B (K x N)
float* mult_thread(int M, const float* A, const float* B, int lda, int ldb, int ldc) {

    const int n = 8; //number of threads
    int m = ldb / n; //int m = M / n;
    thread t[n];
    float* C = new float[M * ldc];

    for (int i = 0; i < n; i++)
        //t[i] = thread( [&, i](){ mult(m, lda, ldc, A + i * m * lda, B, C + i * m * ldc, lda, ldb, ldc);} );
        t[i] = thread([&, i]() { mult(M, lda, m, A, B + i * m, C + i * m, lda, ldb, ldc); });

    for (int i = 0; i < n; i++)
        t[i].join();

    return C;
}

// Zero initialization of the block (M x N) in the matrix
void init_c(int M, int N, float* C, int ldc){
     for (int i = 0; i < M; ++i, C += ldc)
         for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(C + j, _mm256_setzero_ps());
}

// Reordering of (K x 16) block of B
void reorder(int K, const float* B, int ldb, float* B_tmp){
    for(int k = 0; k < K; ++k, B += ldb, B_tmp += 16){
        _mm256_storeu_ps(B_tmp + 0, _mm256_loadu_ps(B + 0));
        _mm256_storeu_ps(B_tmp + 8, _mm256_loadu_ps(B + 8));
    }
}

// Multiplication of (6 x K) block of A and (K x 16) block of B (B - reordered) and
//storeing it to(6 x 16) block in C
void kernel(int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc){
    __m256 a0, a1, b0, b1;
   
    __m256 c00 = _mm256_setzero_ps(); __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(); __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(); __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(); __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(); __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(); __m256 c51 = _mm256_setzero_ps();
    
    
    const int offset0 = lda * 0; const int offset3 = lda * 3;
    const int offset1 = lda * 1; const int offset4 = lda * 4;
    const int offset2 = lda * 2; const int offset5 = lda * 5;
    
    for (int k = 0; k < K; k++){
        b0 = _mm256_loadu_ps(B + 0);
        b1 = _mm256_loadu_ps(B + 8);
        
        a0 = _mm256_broadcast_ss(A + offset0); a1 = _mm256_broadcast_ss(A + offset1);
        
        c00 = _mm256_fmadd_ps(a0, b0, c00); c10 = _mm256_fmadd_ps(a1, b0, c10);
        c01 = _mm256_fmadd_ps(a0, b1, c01); c11 = _mm256_fmadd_ps(a1, b1, c11);
        
        a0 = _mm256_broadcast_ss(A + offset2); a1 = _mm256_broadcast_ss(A + offset3);
        
        c20 = _mm256_fmadd_ps(a0, b0, c20); c30 = _mm256_fmadd_ps(a1, b0, c30);
        c21 = _mm256_fmadd_ps(a0, b1, c21); c31 = _mm256_fmadd_ps(a1, b1, c31);
        
        a0 = _mm256_broadcast_ss(A + offset4); a1 = _mm256_broadcast_ss(A + offset5);
       
        c40 = _mm256_fmadd_ps(a0, b0, c40); c50 = _mm256_fmadd_ps(a1, b0, c50);
        c41 = _mm256_fmadd_ps(a0, b1, c41); c51 = _mm256_fmadd_ps(a1, b1, c51);
        
        B += ldb; A++;
    }
     _mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
     _mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
     C += ldc;
     _mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
     _mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
     C += ldc;
     _mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
     _mm256_storeu_ps(C + 8, _mm256_add_ps(c21, _mm256_loadu_ps(C + 8)));
     C += ldc;
     _mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
     _mm256_storeu_ps(C + 8, _mm256_add_ps(c31, _mm256_loadu_ps(C + 8)));
     C += ldc;
     _mm256_storeu_ps(C + 0, _mm256_add_ps(c40, _mm256_loadu_ps(C + 0)));
     _mm256_storeu_ps(C + 8, _mm256_add_ps(c41, _mm256_loadu_ps(C + 8)));
     C += ldc;
     _mm256_storeu_ps(C + 0, _mm256_add_ps(c50, _mm256_loadu_ps(C + 0)));
     _mm256_storeu_ps(C + 8, _mm256_add_ps(c51, _mm256_loadu_ps(C + 8)));
}

// Product of matrices A (M x K) and B (K x N)
void mult(int M, int K, int N, const float* A, const float* B, float* C, int lda, int ldb,int ldc){
    float* B_tmp = new float[K * 16];
    
    for (int j = 0; j < N; j += 16){
        reorder(K, B + j, ldb, B_tmp);
        for (int i = 0; i < M; i += 6){
             init_c(6, 16, C + i * ldc + j, ldc);
             kernel(K, A + i * lda, B_tmp, C + i * ldc + j, lda, 16, ldc);
        }
    }
    delete[] B_tmp;
}

long Get_Time() {
    using chrono::high_resolution_clock;
    auto t = high_resolution_clock::now();
    auto nanosec = t.time_since_epoch();
    return nanosec.count() / 1000000;
    
}

float* Random(size_t n) {
    float* ret = new float[n];
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution <> dis(0, 1);
   
    for (int i = 0; i < n; i++)
    ret[i] = dis(gen);
   
    return ret;
}
