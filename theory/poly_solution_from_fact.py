import numpy as np

# block S corresponds to the product
def solution_from_factorization_s_one_block(terms, s_idx, variables):
    mask = np.full(terms[0].shape, False, dtype=bool)
    mask[:, :] = False
    mask[s_idx] = True

    terms_i = [i for i in range(len(terms)) if np.any(terms[i][mask] > 0) and np.all(terms[i][~mask] == 0)]
    s = np.sum([terms[i][s_idx] for i in terms_i], axis=0)
    
    return {
        variables[0]: s[0,0],
        variables[1]: s[0,5],
        variables[2]: s[4,0],
    }

# block P corresponds to the linear combination 
def solution_from_factorization_p_one_block(terms, p_idx, variables):
    mask = np.full(terms[0].shape, False, dtype=bool)
    mask[:, :] = False
    mask[p_idx] = True

    terms_i = [i for i in range(len(terms)) if np.any(terms[i][mask] > 0) and np.all(terms[i][~mask] == 0)]
    p = np.sum([terms[i][p_idx] for i in terms_i], axis=0)
    
    # if p is a scalar, then we deal with the linear combination of zero terms, so just record the value of p(which must be zero)
    if len(p.shape) == 0: 
        result = {
             "_L": p
        }
        return result
    
    # empty string as a variable name indicates free coefficient (no actual variable, dummy "variable" which equals to 1)
    result = { var:val for var, val in zip(variables, p[:, 0]) if var != "" }
    result["_L"] = p[len(variables), 0]
    
    return result

def solution_from_factorization_s_blocks(terms, fact_data):
    results = {}
    s_idxs = fact_data["positive"]["s"]["idxs"] + fact_data["negative"]["s"]["idxs"] 
    s_var_blocks = fact_data["positive"]["s"]["expanded_vars"] + fact_data["negative"]["s"]["expanded_vars"] 
    
    for s_idx, s_vars in zip(s_idxs, s_var_blocks):
        results_from_one_block = solution_from_factorization_s_one_block(terms, s_idx, s_vars)
        results = {**results, **results_from_one_block}
    
    return results


def solution_from_factorization_p_blocks(terms, fact_data):
    results = {}
    p_idxs = [fact_data["positive"]["p"]["idx"], fact_data["negative"]["p"]["idx"]]
    p_var_blocks = [fact_data["positive"]["p"]["vars"], fact_data["negative"]["p"]["vars"]]
    
    for p_idx, p_vars in zip(p_idxs, p_var_blocks):
        results_from_one_block = solution_from_factorization_p_one_block(terms, p_idx, p_vars)
        results = {**results, **results_from_one_block}
    
    return results

def solution_from_factorization(terms, fact_data):
    terms = np.array(terms)
    
    results_from_s_blocks = solution_from_factorization_s_blocks(terms, fact_data)
    results_from_p_blocks = solution_from_factorization_p_blocks(terms, fact_data)
    return {**results_from_s_blocks, **results_from_p_blocks}