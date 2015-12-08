/*
   Fast Artificial Neural Network Library (fann)
   Copyright (C) 2003-2012 Steffen Nissen (sn@leenissen.dk)

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <stdio.h>

#include "fann.h"

int main()
{
	fann_type *calc_out;
	unsigned int i;
	int ret = 0;
	int max_expected_idx=0,max_predicted_idx=0,count=0;

	struct fann *ann;
	struct fann_train_data *data;

	printf("Creating network.\n");

#ifdef FIXEDFANN
	ann = fann_create_from_file("mnist_fixed1.net");
#else
	ann = fann_create_from_file("mnist_float.net");
#endif

	if(!ann)
	{
		printf("Error creating ann --- ABORTING.\n");
		return -1;
	}

	fann_print_connections(ann);
	fann_print_parameters(ann);

	printf("Testing network.\n");

#ifdef FIXEDFANN
	data = fann_read_train_from_file("mnist_test.data");
#else
	data = fann_read_train_from_file("mnist_test.data");
#endif

	for(i = 0; i < fann_length_train_data(data); i++)
	{
		fann_reset_MSE(ann);
		calc_out = fann_test(ann, data->input[i], data->output[i]);
		max_expected_idx = 0;
		max_predicted_idx = 0;
		for(int k=1;k<10;k++)
		{
			if(data->output[i][max_expected_idx] < data->output[i][k])
			{
				max_expected_idx = k;
			}
			if(calc_out[max_predicted_idx] < calc_out[k])
			{
				max_predicted_idx = k;
			}
		}

		printf("MNIST test %d  Expected %d , returned=%d\n",
				i,max_expected_idx, max_predicted_idx);
		if(max_expected_idx == max_predicted_idx)
			count++;
	}

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);
	printf("Number correct=%d\n",count);

	return ret;
}
