
/**************************************************************************
 * Desc: Global map storage functions
 * Author: Andrew Howard
 * Date: 6 Feb 2003
 * CVS: $Id: map_store.c,v 1.8 2005/08/19 00:48:20 gerkey Exp $
**************************************************************************/

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#include <libplayercore/error.h>

#include "map.h"


////////////////////////////////////////////////////////////////////////////
// Load an occupancy grid
int map_load_occ(map_t *map, const char *filename, double scale, int negate)
{
  FILE *file;
  char magic[3];
  int i, j;
  int ch, occ;
  int width, height, depth;
  map_cell_t *cell;

  // Open file
  file = fopen(filename, "r");
  if (file == NULL)
  {
    fprintf(stderr, "%s: %s\n", strerror(errno), filename);
    return -1;
  }

  // Read ppm header
  fscanf(file, "%10s \n", magic);
  if (strcmp(magic, "P5") != 0)
  {
    fprintf(stderr, "incorrect image format; must be PGM/binary");
    return -1;
  }

  // Ignore comments
  while ((ch = fgetc(file)) == '#')
    while (fgetc(file) != '\n');
  ungetc(ch, file);

  // Read image dimensions
  fscanf(file, " %d %d \n %d \n", &width, &height, &depth);

  // Allocate space in the map
  if (map->cells == NULL)
  {
    map->scale = scale;
    map->size_x = width;
    map->size_y = height;
    map->cells = (map_cell_t*) calloc(width * height, sizeof(map->cells[0]));
  }
  else
  {
    if (width != map->size_x || height != map->size_y)
    {
      printf("map dimensions are inconsistent with prior map dimensions");
      return -1;
    }
  }

  // Read in the image
  for (j = height - 1; j >= 0; j--)
  {
    for (i = 0; i < width; i++)
    {
      ch = fgetc(file);

      // Black-on-white images
      if (!negate)
      {
        if (ch < depth / 4)
          occ = 1;
        else if (ch > 3 * depth / 4)
          occ = -1;
        else
          occ = 0;
      }

      // White-on-black images
      else
      {
        if (ch < depth / 4)
          occ = -1;
        else if (ch > 3 * depth / 4)
          occ = 1;
        else
          occ = 0;
      }

      if (!MAP_VALID(map, i, j))
        continue;
      cell = map->cells + MAP_INDEX(map, i, j);
      cell->occ_state = occ;
    }
  }

  fclose(file);

  return 0;
}

////////////////////////////////////////////////////////////////////////////
// Save an occupancy grid
int map_save_occ(map_t *map, const char *filename, double scale, int depth)
{
  FILE *file;
  int i, j;
  map_cell_t *cell;

  // Open file
  file = fopen(filename, "w");
  if (file == NULL)
  {
    fprintf(stderr, "%s: %s\n", strerror(errno), filename);
    return -1;
  }

  // Write ppm header
  fprintf(file, "P5 \n");

  // Read image dimensions
  fprintf(file, "%d %d \n%d \n", map->size_x,
		  map->size_y, depth);

  // Read in the image
  for (j = map->size_y - 1; j >= 0; j--)
  {
    for (i = 0; i < map->size_x; i++)
    {
      cell = map->cells + MAP_INDEX(map, i, j);

//      if( i > 235 && i < 260 && j < 159 && j > 134)
//         		  printf("LIBERA, METTO %d\n", cell->occ_state);

      if( cell->occ_state == -1 )
      {
    	  fputc( 255, file);
      }
      else
      {
    	  if( cell->occ_state == 1)
    		  fputc( 0, file);
    	  else
    		  fputc( 127, file);
      }
    }
  }

  fclose(file);

  return 0;
}

////////////////////////////////////////////////////////////////////////////
// Save an occupancy grid
int map_save(map_t *map, const char *filename, double scale, int depth)
{
  FILE *file;
  int i, j;
  map_cell_t *cell;

  // Open file
  file = fopen(filename, "w");
  if (file == NULL)
  {
    fprintf(stderr, "%s: %s\n", strerror(errno), filename);
    return -1;
  }

  // Write ppm header
  fprintf(file, "P5 \n");

  // Read image dimensions
  fprintf(file, "%d %d \n%d \n", map->size_x,
		  map->size_y, 255);

  // Read in the image
  for (j = map->size_y - 1; j >= 0; j--)
  {
    for (i = 0; i < map->size_x; i++)
    {
      cell = map->cells + MAP_INDEX(map, i, j);

//      if( i > 235 && i < 260 && j < 159 && j > 134)
//         		  printf("LIBERA, METTO %d\n", cell->occ_state);
      fputc( cell->occ_state*255/depth, file);

    }
  }

  fclose(file);

  return 0;
}

////////////////////////////////////////////////////////////////////////////
// Save an wifi signal level grid
int map_save_wifi(map_t *map, const char *filename, double scale, int depth)
{
  FILE *file;
  int i, j, k;
  map_cell_t *cell;

  char filenameAp[MAP_WIFI_MAX_LEVELS][255];

  printf("map->nWifiLevel: %d\n", map->nWifiLevel);

  for( k = 0; k < map->nWifiLevel; k++ )
  {
	  sprintf(filenameAp[k], "%s%d.pgm", filename, k);
	  printf("Saving %s...\n", filenameAp[k]);
	  // Open file
	  file = fopen(filenameAp[k], "w");
	  if (file == NULL)
	  {
		fprintf(stderr, "%s: %s\n", strerror(errno), filename);
		return -1;
	  }

	  // Write ppm header
	  fprintf(file, "P5 \n");

	  // Read image dimensions
	  fprintf(file, "%d %d \n%d \n", map->size_x,
			  map->size_y, 255);

	  // Read in the image
	  for (j = map->size_y - 1; j >= 0; j--)
	  {
		for (i = 0; i < map->size_x; i++)
		{
		  cell = map->cells + MAP_INDEX(map, i, j);

	//      if( i > 235 && i < 260 && j < 159 && j > 134)
	//         		  printf("LIBERA, METTO %d\n", cell->occ_state);
		  fputc( cell->wifi_levels[k] *255/depth, file);
		}
	  }

	  fclose(file);
	  printf("Saved %s.\n", filenameAp[k]);
	  file = NULL;
  }
  return 0;
}

int map_save_pathPlanning(map_t *map, const char *filename, double scale, int depth)
{
  FILE *file;
  int i, j;
  map_cell_t *cell;

  // Open file
  file = fopen(filename, "w");
  if (file == NULL)
  {
    fprintf(stderr, "%s: %s\n", strerror(errno), filename);
    return -1;
  }

  // Write ppm header
  fprintf(file, "P5 \n");

  // Read image dimensions
  fprintf(file, "%d %d \n%d \n", map->size_x,
		  map->size_y, 255);

  // Read in the image
  for (j = map->size_y - 1; j >= 0; j--)
  {
    for (i = 0; i < map->size_x; i++)
    {
      cell = map->cells + MAP_INDEX(map, i, j);

//      if( i > 235 && i < 260 && j < 159 && j > 134)
//         		  printf("LIBERA, METTO %d\n", cell->occ_state);
//      if( cell->weight < 5 )
    	  fputc( cell->weight*255/depth, file);
//      else
//    	  fputc( 255, file);

    }
  }

  fclose(file);

  return 0;
}

////////////////////////////////////////////////////////////////////////////
// Load a wifi signal strength map
int map_load_wifi(map_t *map, const char *filename, int index)
{
  FILE *file;
  char magic[3];
  int i, j;
  int ch, level;
  int width, height, depth;
  map_cell_t *cell;

  // Open file
  file = fopen(filename, "r");
  if (file == NULL)
  {
    fprintf(stderr, "%s: %s\n", strerror(errno), filename);
    return -1;
  }

  // Read ppm header
  fscanf(file, "%10s \n", magic);
  if (strcmp(magic, "P5") != 0)
  {
    fprintf(stderr, "incorrect image format; must be PGM/binary");
    return -1;
  }

  // Ignore comments
  while ((ch = fgetc(file)) == '#')
    while (fgetc(file) != '\n');
  ungetc(ch, file);

  // Read image dimensions
  fscanf(file, " %d %d \n %d \n", &width, &height, &depth);

  // Allocate space in the map
  if (map->cells == NULL)
  {
    map->size_x = width;
    map->size_y = height;
    map->cells = (map_cell_t*) calloc(width * height, sizeof(map->cells[0]));
  }
  else
  {
    if (width != map->size_x || height != map->size_y)
    {
      printf("map dimensions are inconsistent with prior map dimensions");
      return -1;
    }
  }

  // Read in the image
  for (j = height - 1; j >= 0; j--)
  {
    for (i = 0; i < width; i++)
    {
      ch = fgetc(file);

      if (!MAP_VALID(map, i, j))
        continue;

      if (ch == 0)
        level = 0;
      else
        level = ch * 100 / 255 - 100;

      cell = map->cells + MAP_INDEX(map, i, j);
      cell->wifi_levels[index] = level;
    }
  }

  fclose(file);

  return 0;
}

unsigned int map_get_pgm(map_t *self, unsigned char **data)
{
	int i, j;
	unsigned char d = 0;
	map_cell_t *cell;

	char tmp[255];
	sprintf(tmp, "P5 %d %d 255\n", self->size_x, self->size_y);

	(*data) = (unsigned char *) calloc(self->size_x * self->size_y + strlen(tmp), sizeof(unsigned char));

	for( i = 0; i < (int)strlen(tmp); i++)
		(*data)[i] = tmp[i];

	// Read in the image
	for (j = 0; j < self->size_y; j++)
	{
		for (i = 0; i < self->size_x; i++)
		{
			cell = self->cells + MAP_INDEX(self, i, self->size_y - 1 - j);

			switch(cell->occ_state)
			{
				case -1: d = 255;
					break;
				case 0: d = 127;
					break;
				case 1: d = 0;
					break;
			}
			(*data)[MAP_INDEX(self, i, j) + strlen(tmp)] = d;
		}
	}


	return (unsigned int) self->size_x *  self->size_y + strlen(tmp);
}

// load wifi map previously saved with map_save_wifi
int map_load_wifi_from_pgm(map_t *map, const char *filename, int index)
{
  FILE *file;
  char magic[3];
  int i, j;
  int ch, level;
  int width, height, depth;
  map_cell_t *cell;

  if( index >= MAP_WIFI_MAX_LEVELS )
  {
	fprintf(stderr, "Too many wifi map (max = %d)\n", MAP_WIFI_MAX_LEVELS);
	return -1;
  }

  // Open file
  file = fopen(filename, "r");
  if (file == NULL)
  {
    fprintf(stderr, "%s: %s\n", strerror(errno), filename);
    return -1;
  }

  // Read ppm header
  fscanf(file, "%10s \n", magic);
  if (strcmp(magic, "P5") != 0)
  {
    fprintf(stderr, "incorrect image format; must be PGM/binary");
    return -1;
  }

  // Ignore comments
  while ((ch = fgetc(file)) == '#')
    while (fgetc(file) != '\n');
  ungetc(ch, file);

  // Read image dimensions
  fscanf(file, " %d %d \n %d \n", &width, &height, &depth);

  // Read in the image
  for (j = height - 1; j >= 0; j--)
  {
    for (i = 0; i < width; i++)
    {
      ch = fgetc(file);

      if (!MAP_VALID(map, i, j))
        continue;

      if (ch == 0)
        level = -100;
      else
        level = -ch;

      cell = map->cells + MAP_INDEX(map, i, j);
      cell->wifi_levels[index] = level;
    }
  }

  fclose(file);

  return 0;
}


