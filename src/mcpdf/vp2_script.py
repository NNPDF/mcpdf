from validphys.loader import Loader
from validphys.convolution import dis_predictions
from validphys.fkparser import load_fktable

l = Loader()
pdf = l.check_pdf('210713-n3fit-001')
ds = l.check_dataset('HERACOMBCCEP', theoryid=200)
table = load_fktable(ds.fkspecs[0])

dis_predictions(table,pdf)