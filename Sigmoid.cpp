#ifndef SIGMOID
#define SIGMOID
#include "Layer.cpp"
#include<cmath>
class Sigmoid : public Hidden_layer
{
public:
	Sigmoid(int features) :Hidden_layer(features, features) {}
	void forward_once() {
		if (output_data == nullptr)output_data = new double[out_features];
		for (int i = 0; i < out_features; i++) {
			output_data[i] = 1.0 / (exp(-input_data[i]) + 1);
		}
		output->set_input_pointer(output_data);
		output->forward_once();
	}
	void backward_once(double* loss) {
		double* next_loss = new double[in_features];
		for (int i = 0; i < in_features; i++)
		{
			next_loss[i] = output_data[i] * (1.0 - output_data[i]) * loss[i];
		}
		input->backward_once(next_loss);
		delete[]next_loss;
		//next_loss = nullptr;
	}
};

#endif