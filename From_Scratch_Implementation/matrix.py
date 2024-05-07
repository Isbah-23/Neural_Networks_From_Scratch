# ill just create my own numpy :) - calling it matrix tho
# making this here because its too crowded otherwise
# and obviously its slower cause its in python list instead of c mats
from math import inf, e, log

class Matrix():
    def __init__(self, two_dim_list):
        if isinstance(two_dim_list, list):
            self.matrix = two_dim_list.copy()
        elif isinstance(two_dim_list, Matrix):
            self.matrix = two_dim_list.matrix.copy()
        else:
            raise ValueError(f"Input must be a list or a Matrix object but it actually was ... (drum roll pls)... {type(two_dim_list)}")

    def T(self):
        return Matrix([[row[column] for row in self.matrix] for column in range(len(self.matrix[0]))])
    
    def argmax(self):
        res = []
        for row in self.matrix:
            max_val, max_idx = -inf, -1
            for i, val in enumerate(row):
                if val > max_val:
                    max_val, max_idx = val, i
            res.append(max_idx)
        return Matrix(res)

    def col_sum_arr(self):
        sum_arr = []
        for i in range(len(self.matrix[0])): # for each col
            total = 0
            for j in range(len(self.matrix)): # for each row
                total += self.matrix[j][i]
            sum_arr.append(total)
        sum_arr = [sum_arr]
        return Matrix(sum_arr)
    
    def row_sum_arr(self):
        sum_arr = []
        for i in range(len(self.matrix)): # for each row
            total = 0
            for j in range(len(self.matrix[0])): # for each col
                total += self.matrix[i][j]
            sum_arr.append(total)
        return Matrix(sum_arr)
    
    def exp(self):
        if isinstance(self.matrix[0], list):
            return Matrix([[e**elem for elem in row] for row in self.matrix])
        elif isinstance(self.matrix[0],(int. float)):
            return Matrix([e**elem for elem in self.matrix])

    def dot(self,m2):
        m3 = []
        if isinstance(m2,list):
            if len(self.matrix[0]) == len(m2):
                for i in range(len(self.matrix)):
                    m3.append([])
                    for j in range(len(m2[0])):
                        total = 0
                        for k in range(len(m2)):
                            total += self.matrix[i][k]*m2[k][j]
                        m3[i].append(total)
                return Matrix(m3)
            else:
                raise ValueError("Check your mat dims einstein")
        elif isinstance(m2,Matrix):
            if len(self.matrix[0]) == len(m2.matrix):
                for i in range(len(self.matrix)):
                    m3.append([])
                    for j in range(len(m2.matrix[0])):
                        total = 0
                        for k in range(len(m2.matrix)):
                            total += self.matrix[i][k]*m2.matrix[k][j]
                        m3[i].append(total)
                return Matrix(m3)
            else:
                raise ValueError("Check your mat dims einstein")

    def __add__(self, m2):
        if isinstance(m2,Matrix) and type(m2.matrix[0]) == list:
            if len(self.matrix) == len(m2.matrix) and len(self.matrix[0]) == len(m2.matrix[0]):
                return Matrix([[self.matrix[i][j]+m2.matrix[i][j] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            elif len(self.matrix) != len(m2.matrix) and len(m2.matrix) == 1 and len(self.matrix[0]) == len(m2.matrix[0]):
                return Matrix([[self.matrix[i][j]+m2.matrix[0][j] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        elif isinstance(m2,Matrix) and type(m2.matrix[0]) == int and isinstance(self.matrix[0],list):
            if len(self.matrix) == len(m2.matrix):
                return Matrix([[self.matrix[i][j]+m2.matrix[i] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        elif isinstance(m2, list):
            if len(self.matrix) == len(m2):
                return Matrix([[self.matrix[i][j]+m2[i] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        elif isinstance(m2, Matrix) and isinstance(m2.matrix[0],(int,float)) and isinstance(self.matrix[0],(int,float)):
            if len(self.matrix) == len(m2.matrix):
                return Matrix([self.matrix[i]+m2.matrix[i] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
            
    def __sub__(self, m2):
        if isinstance(m2, Matrix) and type(m2.matrix[0]) == list:
            if len(self.matrix) == len(m2.matrix) and len(self.matrix[0]) == len(m2.matrix[0]):
                return Matrix([[self.matrix[i][j] - m2.matrix[i][j] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        elif isinstance(m2, Matrix) and isinstance(m2.matrix[0],(int,float)):
            if len(self.matrix) == len(m2.matrix):
                return Matrix([[self.matrix[i][j] - m2.matrix[i] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        elif isinstance(m2, list):
            if len(self.matrix) == len(m2):
                return Matrix([[self.matrix[i][j] - m2[i] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        else:
            print(f"Type aint supported chief")
    
    def __truediv__(self, m2):
        if isinstance(m2, Matrix) and type(m2.matrix[0]) == list:
            if len(self.matrix) == len(m2.matrix) and len(self.matrix[0]) == len(m2.matrix[0]):
                return Matrix([[self.matrix[i][j] / m2.matrix[i][j] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        elif isinstance(m2, Matrix) and isinstance(m2.matrix[0],(int,float)):
            if len(self.matrix) == len(m2.matrix):
                return Matrix([[self.matrix[i][j] / m2.matrix[i] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        elif isinstance(m2, list):
            if len(self.matrix) == len(m2):
                return Matrix([[self.matrix[i][j] / m2[i] for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
            else:
                raise ValueError("Incorrect mat dims...")
        elif isinstance(m2, int):
            return Matrix([[self.matrix[i][j] / m2 for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
        elif isinstance(m2, int):
            return Matrix([[self.matrix[i][j] / m2 for j in range(len(self.matrix[0]))] for i in range(len(self.matrix))])
        else:
            print(f"Type aint supported chief")
    
    def max(self):
        res = []
        for row in self.matrix:
            max_val= -inf
            for i in range(len(row)):
                if row[i] > max_val:
                    max_val = row[i]
            res.append(max_val)
        return Matrix(res)

    def __neg__(self):
        if isinstance(self.matrix[0], list):
            return Matrix([[-elem for elem in row] for row in self.matrix])
        elif isinstance(self.matrix[0], (int,float)):
            return Matrix([-elem for elem in self.matrix])
    
    def clip(self, lower_lim, upper_lim):
        if isinstance(self.matrix[0], list):
            partial =  Matrix([[max(elem,lower_lim) for elem in row] for row in self.matrix])
            return Matrix([[min(elem,upper_lim) for elem in row] for row in partial.matrix])
        elif isinstance(self.matrix[0], (int,float)):
            partial =  Matrix([max(elem,lower_lim) for elem in self.matrix])
            return Matrix([min(elem,upper_lim) for elem in partial.matrix])
    
    def __repr__(self):
        def format_float(item, check):
            if check:
                return "{:.8e}".format(item)
            else:
                return "{:.8f}".format(item)
        def any_less(val):
            if isinstance(self.matrix[0],list):
                for row in self.matrix:
                    for elem in row:
                        if abs(elem) < val:
                            return True
                return False
            if isinstance(self.matrix[0],(int,float)):
                for elem in self.matrix:
                    if abs(elem) < val:
                            return True
                return False
        check = any_less(1e-4)
        if isinstance(self.matrix[0], list):
            max_width = max(len(format_float(float(item), check)) for row in self.matrix for item in row if isinstance(item, (float, int)))
            rows = []
            for row in self.matrix:
                row_str = []
                for item in row:
                    if isinstance(item, float):
                        row_str.append(format_float(item, check).rjust(max_width))
                    else:
                        row_str.append(str(item))
                rows.append(" ".join(row_str))
            return "[[" + "]\n [".join(rows) + "]]"
        elif isinstance(self.matrix[0],(int, float)):
            max_width = max(len(format_float(item, check)) for item in self.matrix)
            rows = []
            for item in self.matrix:
                row_str = []
                if isinstance(item, float):
                    row_str.append(format_float(item, check).rjust(max_width))
                else:
                    row_str.append(str(item))
                rows.append(" ".join(row_str))
            return "[[" + "]\n [".join(rows) + "]]"
        
    def log(self):
        if isinstance(self.matrix[0], list):
            return Matrix([[log(elem,e) for elem in row] for row in self.matrix])
        elif isinstance(self.matrix[0],(int, float)):
            return Matrix([log(elem,e) for elem in self.matrix])
    
    def mean(self): # ig this is only called for 1d in nnfs so am implementing only for that
        total = 0
        for elem in self.matrix:
            total += elem
        return total/len(self.matrix)
    
    def acc(self, y_true): # is only called for 1D I hope
        return Matrix([elem==y_true[i] for i,elem in enumerate(self.matrix)])
    
    def __mul__(self, m2):
        if isinstance(m2, (int, float)):
            if isinstance(self.matrix[0], list):
                return Matrix([[elem * m2 for elem in row] for row in self.matrix])
            elif isinstance(self.matrix[0], (int, float)):
                return Matrix([elem * m2 for elem in self.matrix])
        else:
            raise TypeError(f"called for {self.matrix} and {type(m2)}")