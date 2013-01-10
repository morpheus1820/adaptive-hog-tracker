#include <iostream>
#include <stdio.h>
#include "map.h"

int main(int argc, char *argv[])
{
	using namespace std;

	map_t *map;
	char filename[] = "magazzino3b.pgm";

	map = map_alloc();

	if( map_load_occ(map, filename, 0.1, 0) < 0)
	{
		cout << "ERRORE!!!!" << endl;
		return -1;
	}

	cout << "OK" << endl;
	cout << map->size_x << " " << map->size_y << endl;
	
	map_update_cspace(map, 5.0);

	FILE* f;
	f = fopen("pippo","w");

	for (int j = map->size_y - 1; j >= 0; j--)
  	{
		for (int i = 0; i < map->size_x; i++)
		{
			map_cell_t* cell = map->cells + MAP_INDEX(map, i, j);
			if( cell->occ_state < 0 )
				fprintf(f,"|%3.3f", cell->occ_dist);
			else
				if( cell->occ_state == 0 )
					fprintf(f,"!%3.3f", cell->occ_dist);
				else
					fprintf(f,"|%3.3f", cell->occ_dist);
// 			cout << " " << cell->occ_state << " " << cell->occ_dist;
		}
		fprintf(f, "\n");
	}

	return 0;
}
