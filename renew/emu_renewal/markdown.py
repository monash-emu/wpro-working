import pandas as pd

def get_param_table_from_instrum(instrum, dec_places=2):
    param_table = pd.DataFrame(columns=['Starting value', 'Lower bound', 'Upper bound'])
    for param in instrum[0]:
        start_val = param.value if param.dimension == 1 else param.value[0]
        param_table.loc[param.name] = {
            'Starting value': round(start_val, dec_places), 
            'Lower bound': round(param.bounds[0][0], dec_places), 
            'Upper bound': round(param.bounds[1][0], dec_places),
        }
    return param_table