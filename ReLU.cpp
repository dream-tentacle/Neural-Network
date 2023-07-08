#ifndef RELU
#define RELU
#include "Layer.cpp"
class ReLU :public Hidden_layer
{
public:
	ReLU(int features) :Hidden_layer(features, features) {}
	void forward_once() {
		if (output_data == nullptr)output_data = new double[out_features];
		for (int i = 0; i < out_features; i++) {
			output_data[i] = input_data[i] >= 0 ? input_data[i] : 0;
		}
		output->set_input_pointer(output_data);
		output->forward_once();
	}
	void backward_once(double* loss) {
		double* next_loss = new double[in_features];
		for (int i = 0; i < in_features; i++)
		{
			next_loss[i] = input_data[i] > 0 ? loss[i] : 0;
		}
		input->backward_once(next_loss);
		delete[]next_loss;
		//next_loss = nullptr;
	}
};
#endif