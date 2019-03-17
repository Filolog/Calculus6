# -*- coding: utf-8 -*-
import numpy as np
from matrix import Matrix
from matrix import Norm
import copy
import time


def solveCramer(A, b):
    det = A.det()
    m = len(A.rows[0])
    res = []
    
    for i in range(0, m):
        tmp = A.copy()
        tmp.replaceColumn(b.getColumn(0), i)
        res.append([tmp.det() / det])
    return Matrix.fromList(res)    


def condNumber(A):
    return Norm.norm8(A) * Norm.norm8(A.reverse())
    

Report = open('11.html', 'w', encoding="utf-8")

Report.write("<!DOCTYPE html>\n <html>\n  <head>\n   <meta charset=\"utf-8\">\n") 
Report.write("             <title>Вычислительный практикум</title>\n <script\n src=\"http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML\">")
Report.write("</script>\n </head>\n <body>\n")
Report.write("<center><b>1.1 Обусловленность СЛАУ</b> </center>\n")
Report.write("<p><i> 1. Решить по формулам Крамера системы \(Ax = b\), \(A\overline{x} = b + \Delta b\), где: </i>")

A = Matrix.fromList([[-400.60, 199.8],
                    [1198.8, -600.4]])
b = Matrix.fromList([[200], [600]])
db = Matrix.fromList([[-1], [-1]])

Report.write("<br>\(A = "+A.toTexForm() + ", b = " + b.toTexForm() + " ,\Delta b = " + db.toTexForm() + "\) <br>")

Report.write("\(x = " + solveCramer(A, b).toTexForm() + " ,\overline{x} = " + solveCramer(A, b + db).toTexForm() + "\) <br>")

Report.write("<i>Найти число обусловленности \( \mu (A) = ||A|| \cdot || A^{-1}||, \mu (A) = " + "{0:15.3f}".format(condNumber(A)) + "\) <i> <br>")



Report.write("</body>\n</html>")

Report.close()