using Pkg
Pkg.activate(".")
Pkg.add("PyCall")
ENV["PYTHON"] = # enter "which python" folder here
Pkg.build("PyCall")
Pkg.instantiate()