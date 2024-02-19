import MatrixMarket as mm
using SparseArrays

mat = SparseMatrixCSC(mm.mmread("jpwh_991.mtx")');
rowptr = mat.colptr;
colval = mat.rowval;
V = mat.nzval;

open("R.csv", "w") do io
	map(rowptr) do val
	  println(io, val)
	end
end

open("C.csv", "w") do io
	map(colval) do val
	  println(io, val)
	end
end

open("V.csv", "w") do io
	map(V) do val
	  println(io, val)
	end
end