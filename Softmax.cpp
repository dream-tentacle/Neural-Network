#ifndef SOFTMAX
#define SOFTMAX
#include "Layer.cpp"
#include<cmath>
const int CHOOSE_MAX = 1000;
class Softmax_output_layer : public Output_layer
{
public:
	double loss = 0, * loss_gradient;
	double a_exp_sum = 0, min_a = 0;
	Softmax_output_layer(int input_size) :Output_layer(input_size, input_size) {
		loss_gradient = new double[input_size];
		output_data = new double[input_size];
	}
	void forward_once() {
		a_exp_sum = 0;
		min_a = input_data[0];
		for (int i = 1; i < in_features; i++){
			if (min_a > input_data[i])min_a = input_data[i];
		}
		if (min_a < 0)
			min_a = 0;
		for (int i = 0; i < in_features; i++){
			if(input_data[i] - min_a > CHOOSE_MAX)a_exp_sum += exp(CHOOSE_MAX);
			else 
				a_exp_sum += exp(input_data[i] - min_a);
			if (isinf(a_exp_sum))printf("%f\n", input_data[i]);
		}
		if (a_exp_sum <= 0.0001)exit(-2);
		for (int i = 0; i < in_features; i++){
			if (input_data[i] - min_a > CHOOSE_MAX)
				output_data[i] = exp(CHOOSE_MAX) / a_exp_sum;
			else output_data[i] = exp(input_data[i] - min_a) / a_exp_sum;
		}
	}
	void print_output() {
		for (int i = 0; i < out_features; i++)
			printf("%f%c", output_data[i], i == out_features - 1 ? '\n' : ' ');
	}
	void backward_once(double* true_label) {
		if (output_data[int(*true_label)] > 0.000001)
			loss = -log(output_data[int(*true_label)]);
		else loss = 10;
		for (int i = 0; i < in_features; i++){
			if (i == int(*true_label)) {
				loss_gradient[i] = output_data[i] - 1;
			}
			else loss_gradient[i] = output_data[i];
			if (isnan(loss_gradient[i])) {
				printf("%f %f\n", input_data[i],
					a_exp_sum);
				exit(-2);
			}
		}
		input->backward_once(loss_gradient);
	}
	double loss_sum() {
		return loss;
	}
	int get_ans() {
		int re = 0;
		double max_re = output_data[0];
		for (int i = 1; i < out_features; i++){
			if (max_re < output_data[i]) {
				max_re = output_data[i];
				re = i;
			}
		}
		return re;
	}
};
#endif