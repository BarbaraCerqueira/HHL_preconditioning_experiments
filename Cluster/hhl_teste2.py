from linear_solvers import NumPyLinearSolver, HHL
from linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz
from qiskit.quantum_info import Statevector
from scipy.sparse import diags
from scipy.linalg import toeplitz, circulant
import numpy as np
import time

from qiskit.utils import algorithm_globals
algorithm_globals.massive=True

def generate_tridi(a, b, matrix_size, toeplitz=False):
    if not isinstance(matrix_size, int) or np.log2(matrix_size) % 1 != 0:
        raise ValueError("Matrix dimension must be an integer 2^n.")

    if toeplitz:
        tridi_matrix = TridiagonalToeplitz(np.log2(matrix_size), a, b)
    else:
        tridi_matrix = diags([b, a, b], [-1, 0, 1], shape=(matrix_size, matrix_size)).toarray()

    return tridi_matrix

def solve_hhl(matrix, vector, tol=1e-2, verbose=False):
    matrix = np.array(matrix, dtype=float)
    vector = np.array(vector, dtype=float)
    
    matrix_old = matrix
    matrix = matrix / np.linalg.norm(vector)
    vector = vector / np.linalg.norm(vector)
    n = matrix.shape[0]
    
    start_time_real = time.time()
    start_time = time.process_time()
    hhl = HHL(tol).solve(matrix, vector)
    elapsed_cpu_time = time.process_time() - start_time
    elapsed_real_time = time.time() - start_time_real

    # Get solution vector from state vector
    result_vector = Statevector(hhl.state).data.real
    start_index = int(result_vector.shape[0]/2)
    result_vector = result_vector[start_index : (start_index + n)]
    result_vector = hhl.euclidean_norm * result_vector / np.linalg.norm(result_vector)

    if verbose:
        print(f"Matrix: {matrix_old}")
        print(f"HHL Solution Norm: {hhl.euclidean_norm}")
        print(f"HHL Solution Vector: {result_vector}")
        print(f"CPU Time: {elapsed_cpu_time} seconds")

    return result_vector, elapsed_cpu_time, elapsed_real_time

def solve_classical(matrix, vector, verbose=False):
    matrix = np.array(matrix, dtype=float)
    vector = np.array(vector, dtype=float)

    matrix = matrix / np.linalg.norm(vector)
    vector = vector / np.linalg.norm(vector)

    classical = NumPyLinearSolver().solve(matrix, vector)
    result_vector = classical.state

    if verbose:
        print(f"Classical Solution Norm: {classical.euclidean_norm}")
        print(f"Classical Solution Vector: {result_vector}")

    return result_vector

def error(classical, hhl):
    return np.linalg.norm(classical - hhl)/ np.linalg.norm(classical)

# Add zeros so that matrix is of size 2^n
def extend_matrix_simple(matrix, vector):
    n = matrix.shape[0]
    
    if np.log2(n) % 1 != 0:
        extend_n = int(2**(np.floor(np.log2(n))+1))
    else:
        extend_n = n
        
    new_matrix = np.eye(extend_n)
    new_matrix[:n,:n] = matrix

    new_vector = np.append(vector, [0]*(extend_n - n))

    return new_matrix, new_vector

# Extend to an hermitian matrix
def extend_matrix_hermitian(matrix, vector):
    if not is_hermitian(matrix):
        n = matrix.shape[0]
        extend_n = 2*n

        new_matrix = np.zeros([extend_n, extend_n])
        new_matrix[:n,n:] = np.conj(matrix.T)
        new_matrix[n:,:n] = matrix
        
        new_vector = np.append(np.conj(vector), vector)
        return new_matrix, new_vector
    else:
        return matrix, vector

# Verify if matrix is hermitian
def is_hermitian(matrix):
    return np.allclose(matrix, np.conj(matrix.T))

# Modify matrix and vector so that they can be used on HHL (hermitian and of dimension 2^n)
def set_system(matrix,vector):
    matrix = np.array(matrix, dtype=float)
    vector = np.array(vector, dtype=float)
    n = matrix.shape[0]
    
    if np.log2(n) % 1 != 0: # Matrix is not of size 2^n
        matrix, vector = extend_matrix_simple(matrix, vector)
    if not is_hermitian(matrix): # Matrix is not hermitian
        matrix, vector = extend_matrix_hermitian(matrix,vector)   
    return matrix, vector

def condition_number(matrix):
    matrix = np.array(matrix, dtype=float)
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues_abs = np.abs(eigenvalues)
    eigen_max = np.max(eigenvalues_abs)
    eigen_min = np.min(eigenvalues_abs)
    return eigen_max/eigen_min

# Solve system without preconditioner and print info (matrix/vector doesn't need to be extended previously)
def solve_wp(matrix,vector, tol=1e-2):
    matrix = np.array(matrix, dtype=float)
    vector = np.array(vector, dtype=float)  
    n = matrix.shape[0]
    cond = condition_number(matrix)
    
    set_matrix1,set_vector1 = set_system(matrix,vector)
    
    # Non preconditioned system resolution
    hhl_result1, cpu_time1, real_time1 = solve_hhl(set_matrix1,set_vector1,tol)
    classical_result1 = solve_classical(set_matrix1,set_vector1)
    hhl_result1, classical_result1 = hhl_result1[:n], classical_result1[:n]
    result_error1 = error(classical_result1, hhl_result1)
    
    print(f"\nSolving system:")
    if set_matrix1.shape[0] > matrix.shape[0]:
        print(f"Warning: System had to be extended to N = {set_matrix1.shape[0]}. Input matrix is not hermitian and/or not of size 2^n. Condition number of extended matrix is actually {condition_number(set_matrix1)}\n")
    print(f"\nCondition Number: {cond}")
    print(f"HHL Result: {hhl_result1}")
    print(f"Classical Result: {classical_result1}")
    print(f"Error: {result_error1}")
    print(f"CPU Time: {cpu_time1}")
    print(f"Wall Time: {real_time1}")

# Solve system and print info
def solve(matrix,vector,preconditioner,tol=1e-2):
    matrix = np.array(matrix, dtype=float)
    vector = np.array(vector, dtype=float)  
    n = matrix.shape[0]
    preconditioner = np.array(preconditioner, dtype=float)
    cond_before = condition_number(matrix)
    
    pc_matrix = np.dot(np.linalg.inv(preconditioner), matrix)
    pc_vector = np.dot(np.linalg.inv(preconditioner), vector)
    cond_after = condition_number(pc_matrix)
    
    set_matrix1,set_vector1 = set_system(matrix,vector)
    #print(f"Matrix NPC: {set_matrix1} and vector NPC is {set_vector1}") # Debug
    set_matrix2,set_vector2 = set_system(pc_matrix,pc_vector)
    #print(f"Matrix PC: {set_matrix2} and vector PC is {set_vector2}") # Debug

    # Non preconditioned system resolution
    hhl_result1, cpu_time1, real_time1 = solve_hhl(set_matrix1,set_vector1,tol)
    classical_result1 = solve_classical(set_matrix1,set_vector1)
    hhl_result1, classical_result1 = hhl_result1[:n], classical_result1[:n]
    result_error1 = error(classical_result1, hhl_result1)
    
    print(f"\nNon preconditioned system:")
    if set_matrix1.shape[0] > matrix.shape[0]:
        print(f"Warning: System had to be extended to N = {set_matrix1.shape[0]}. Input matrix is not hermitian and/or not of size 2^n. Condition number of extended matrix is actually {condition_number(set_matrix1)}")
    print(f"\nCondition Number: {cond_before}")
    print(f"HHL Result: {hhl_result1}")
    print(f"Classical Result: {classical_result1}")
    print(f"Error: {result_error1}")
    print(f"CPU Time: {cpu_time1}")
    print(f"Wall Time: {real_time1}")
    
    # Preconditioned system resolution
    hhl_result2, cpu_time2, real_time2 = solve_hhl(set_matrix2,set_vector2,tol)
    classical_result2 = solve_classical(set_matrix2,set_vector2)
    hhl_result2, classical_result2 = hhl_result2[:n], classical_result2[:n]
    result_error2 = error(classical_result1, hhl_result2) # Result compared to classical result before preconditioning
    
    print(f"\nPreconditioned system:")
    if set_matrix2.shape[0] > matrix.shape[0]:
        print(f"Warning: System had to be extended to N = {set_matrix2.shape[0]}. Input matrix is not hermitian and/or not of size 2^n. Condition number of extended matrix is actually {condition_number(set_matrix2)}")
    print(f"\nCondition Number: {cond_after}")
    print(f"HHL Result: {hhl_result2}")
    print(f"Classical Result: {classical_result2}")
    print(f"Error: {result_error2}")
    print(f"CPU Time: {cpu_time2}")
    print(f"Wall Time: {real_time2}")

def circulant_preconditioner(toeplitz):
    np.array(toeplitz, dtype=float)
    col = toeplitz[:, 0]
    row = toeplitz[0, :]
    n = toeplitz.shape[0]
    
    col = np.append(col,0)
    i = np.arange(n)
    c = (i * col[n-i] + (n-i) * row[i]) / n
    
    circulant_matrix = circulant(c)
    circulant_matrix = np.array(circulant_matrix, dtype=float)
    return circulant_matrix


if __name__ == "__main__":
    # Example N = 16 (expands to N = 32) - not hermitian
    example = np.array([
    [5, 1, 1, 6, 1, 6, 4, 2, -1, 5, 5, 5, -1, 7, 7, 5],
    [-2, 5, 1, 1, 6, 1, 6, 4, 2, -1, 5, 5, 5, -1, 7, 7],
    [7, -2, 5, 1, 1, 6, 1, 6, 4, 2, -1, 5, 5, 5, -1, 7],
    [3, 7, -2, 5, 1, 1, 6, 1, 6, 4, 2, -1, 5, 5, 5, -1],
    [2, 3, 7, -2, 5, 1, 1, 6, 1, 6, 4, 2, -1, 5, 5, 5],
    [5, 2, 3, 7, -2, 5, 1, 1, 6, 1, 6, 4, 2, -1, 5, 5],
    [3, 5, 2, 3, 7, -2, 5, 1, 1, 6, 1, 6, 4, 2, -1, 5],
    [6, 3, 5, 2, 3, 7, -2, 5, 1, 1, 6, 1, 6, 4, 2, -1],
    [2, 6, 3, 5, 2, 3, 7, -2, 5, 1, 1, 6, 1, 6, 4, 2],
    [5, 2, 6, 3, 5, 2, 3, 7, -2, 5, 1, 1, 6, 1, 6, 4],
    [6, 5, 2, 6, 3, 5, 2, 3, 7, -2, 5, 1, 1, 6, 1, 6],
    [4, 6, 5, 2, 6, 3, 5, 2, 3, 7, -2, 5, 1, 1, 6, 1],
    [-2, 4, 6, 5, 2, 6, 3, 5, 2, 3, 7, -2, 5, 1, 1, 6],
    [6, -2, 4, 6, 5, 2, 6, 3, 5, 2, 3, 7, -2, 5, 1, 1],
    [0, 6, -2, 4, 6, 5, 2, 6, 3, 5, 2, 3, 7, -2, 5, 1],
    [-1, 0, 6, -2, 4, 6, 5, 2, 6, 3, 5, 2, 3, 7, -2, 5]],dtype=float)

    precond_ex = circulant_preconditioner(example)
    vector_ex = [1]*16

    solve(matrix=example, vector=vector_ex, preconditioner=precond_ex, tol=1e-2)



