# Parse a boolean matrix from Prolog version of BMLP which has integers as rows
def parse_prolog_binary_codes(path):
    
    codes = []
    max_length = 0
    
    with open(path, 'r') as prolog:
        
        for row in prolog:
            
            # Get the integer representation of the bitcode
            code, length = integer_to_binary_code(int(row.split('(')[1].split(')')[0]))
            codes.append(code)
            max_length = max(max_length, length)
            
    return codes, max_length

def integer_to_binary_code(n):
    len = n.bit_length()
    return [n >> i & 1 for i in range(0, len)], len