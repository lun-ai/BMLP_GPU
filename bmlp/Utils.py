# Parse a boolean matrix from Prolog version of BMLP which has integers as rows
def parse_prolog_binary_codes(path):
    
    codes = []
    max_length = 0
    
    with open(path, 'r') as prolog:
        
        num_facts = 0
        
        for row in prolog:
            if "%" not in row and row != "\n":
                
                # Get the integer representation of the bitcode
                code, length = integer_to_binary_code(int(row.replace(" ", "").replace("\n", "").split(",")[1].strip(").")))
                codes.append(code)
                max_length = max(max_length, length)
                num_facts += 1
            
    return codes, num_facts, max_length, 

def integer_to_binary_code(n):
    len = n.bit_length()
    return [n >> i & 1 for i in range(0, len)], len