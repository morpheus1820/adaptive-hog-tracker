all:
	g++ -ggdb \
	`pkg-config --cflags --libs opencv` \
	`gsl-config --cflags --libs` \
 	main.cc \
	-lm -o main

clean:
	rm main
	rm -rf *.dSYM
	rm *.o

