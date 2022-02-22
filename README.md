# heat_transfer

This is a school project, which solves the heat equation in 2D using CUDA C++. After a successful computation, the results are stored in the results directory. The number of the text file name is the iteration of the computation. The number in the time.txt file is the amount of milliseconds the computation took.

If you use Linux then to run this program do the following:
* go to the directory
* check the Makefile and make sure that there is correct sm number on the third line
* either make sure that do is executable and type ./do
* or type:

make clean

make main

./main