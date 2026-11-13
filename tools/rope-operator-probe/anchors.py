"""Actual source callback names, including their layer suffix."""
def stage_names(role):
 if role=='q':return 'Qcur_normed-3','Qcur-3',16
 if role=='k':return 'Kcur_normed-3','Kcur-3',4
 raise ValueError('closed Q/K role')
