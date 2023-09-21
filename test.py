import numpy as np
# from visuallization import show_user_img

def minimalize(input_array):
    n=280
    input_array = input_array.reshape(n, n)
    # 입력 배열 (n, n)
    # input_array = np.array([[1, 1, 0, 0],
    #                         [1, 1, 0, 0],
    #                         [0, 0, 0, 0],
    #                         [0, 0, 0, 0]])

    # 출력 배열 (m, m)
    m = 28
    output_array = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            subblock = input_array[i*(n//m):(i+1)*(n//m), j*(n//m):(j+1)*(n//m)]
            output_array[i, j] = int(np.mean(subblock))

    # print(output_array)
    show_user_img(output_array.flatten())
    return output_array.flatten()