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

int FANN_API test_callback(struct fann *ann, struct fann_train_data *train,
	unsigned int max_epochs, unsigned int epochs_between_reports, 
	float desired_error, unsigned int epochs)
{
	printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
	return 0;
}

int main()
{
	fann_type *calc_out;
	const unsigned int num_input = 50;
	const unsigned int num_output = 10;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 100;
	const float desired_error = (const float) 0;
	const unsigned int max_epochs = 100;
	const unsigned int epochs_between_reports = 1;
	struct fann *ann;
	struct fann_train_data *data;

	unsigned int i = 0;
	unsigned int decimal_point;
	int max_expected_idx=0,max_predicted_idx=0,count=0;


	printf("Creating network.\n");
	ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	data = fann_read_train_from_file("mnist_train.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_init_weights(ann, data);
	
	printf("Training network.\n");
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

	printf("Saving network.\n");

	fann_save(ann, "mnist_float.net");

	decimal_point = fann_save_to_fixed(ann, "mnist_fixed.net");
	fann_save_train_to_fixed(data, "mnist_fixed.data", decimal_point);

	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);

	/*Test the training*/
	ann = fann_create_from_file("mnist_float.net");
	data = fann_read_train_from_file("mnist_train.data");
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

	return 0;
}
