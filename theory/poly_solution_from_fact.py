import numpy as np

# block P corresponds to the product constraint
def solution_from_factorization_p_one_block(terms, p_idx, variables):
    mask = np.full(terms[0].shape, False, dtype=bool)
    mask[:, :] = False
    mask[p_idx] = True

    terms_i = [i for i in range(len(terms)) if np.any(terms[i][mask] > 0) and np.all(terms[i][~mask] == 0)]
    p = np.sum([terms[i][p_idx] for i in terms_i], axis=0)
    
    return {
        variables[0]: p[0,0],
        variables[1]: p[0,5],
        variables[2]: p[4,0],
    }

# block S corresponds to the linear combination constraint
def solution_from_factorization_s_one_block(terms, s_idx, variables):
    mask = np.full(terms[0].shape, False, dtype=bool)
    mask[:, :] = False
    mask[s_idx] = True

    terms_i = [i for i in range(len(terms)) if np.any(terms[i][mask] > 0) and np.all(terms[i][~mask] == 0)]
    s = np.sum([terms[i][s_idx] for i in terms_i], axis=0)
    
    # if s is a scalar, then we deal with the linear combination of zero terms, so just record the value of p(which must be zero)
    if len(s.shape) == 0: 
        result = {
             "_L": s
        }
        return result
    
    # empty string as a variable name indicates free coefficient (no actual variable, dummy "variable" which equals to 1)
    result = { var:val for var, val in zip(variables, s[:, 0]) if var != "" }
    result["_L"] = s[len(variables), 0]
    
    return result

def solution_from_factorization_p_blocks(terms, fact_data):
    results = {}
    p_idxs = fact_data["positive"]["p"]["idxs"] + fact_data["negative"]["p"]["idxs"] 
    p_var_blocks = fact_data["positive"]["p"]["expanded_vars"] + fact_data["negative"]["p"]["expanded_vars"] 
    
    for p_idx, p_vars in zip(p_idxs, p_var_blocks):
        results_from_one_block = solution_from_factorization_p_one_block(terms, p_idx, p_vars)
        results = {**results, **results_from_one_block}
    
    return results


def solution_from_factorization_s_blocks(terms, fact_data):
    results = {}
    s_idxs = [fact_data["positive"]["s"]["idx"], fact_data["negative"]["s"]["idx"]]
    s_var_blocks = [fact_data["positive"]["s"]["vars"], fact_data["negative"]["s"]["vars"]]
    
    for s_idx, s_vars in zip(s_idxs, s_var_blocks):
        results_from_one_block = solution_from_factorization_s_one_block(terms, s_idx, s_vars)
        results = {**results, **results_from_one_block}
    
    return results

def solution_from_factorization(terms, fact_data):
    terms = np.array(terms)
    
    results_from_s_blocks = solution_from_factorization_s_blocks(terms, fact_data)
    results_from_p_blocks = solution_from_factorization_p_blocks(terms, fact_data)
    return {**results_from_s_blocks, **results_from_p_blocks}