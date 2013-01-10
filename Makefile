all:
	g++ -ggdb \
	`pkg-config --cflags opencv` `pkg-config --libs opencv` \
	`gsl-config --cflags --libs` \
	main.cc pf/*.c Map/*.c \
	 my_HOG.cpp -lm -o main
