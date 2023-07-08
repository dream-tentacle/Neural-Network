#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>	
#include <Windows.h>
#include "Layer.cpp"
#include "Linear.cpp"
#include "ReLU.cpp"
#include "Softmax.cpp"

std::vector<double> train_dataset,train_labelset,test_dataset,test_labelset;
Input_layer input(28 * 28);
Linear x1(28 * 28, 100);
ReLU relu(100);
Linear x3(100, 10);
Softmax_output_layer output(10);
int cnt = 0;
double rd() {
#define isdigit(p) ((p)<='9'&&(p)>='0')
    char x = getchar();
    double re = 0;
    while (!isdigit(x))x = getchar();
    while (isdigit(x)) {
        re = re * 10 + (x - '0');
        x = getchar();
    }
    if (x == '.') {
        re += 0.1 * (x - '0');
        x = getchar();
        double t = 0.01;
        while (isdigit(x)) {
            re += t * (x - '0');
            x = getchar();
            t *= 0.1;
        }
    }
    return re;
}
void originate() {
    using namespace std;
    //hide the cursor (just for beauty)
    CONSOLE_CURSOR_INFO cursor;
    cursor.bVisible = 0;
    cursor.dwSize = 1;
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleCursorInfo(hOut, &cursor);

    freopen("E:\\code\\c++\\AI\\Neural Network\\MNIST_data\\train_label.txt", "r", stdin);
    for (int i = 0; i < 60000; i++){
        train_labelset.push_back(rd());
    }
    fclose(stdin);
    freopen("E:\\code\\c++\\AI\\Neural Network\\MNIST_data\\test_label.txt", "r", stdin);
    for (int i = 0; i < 10000; i++){
        test_labelset.push_back(rd());
    }
    freopen("E:\\code\\c++\\AI\\Neural Network\\MNIST_data\\test_image.txt", "r", stdin);
    for (int i = 1; i <= 10000; i++)
    {
        for (int j = 0; j < 28 * 28; j++){
            test_dataset.push_back(rd());
        }
        if (i % 100 == 0){
            cout << "\r";
            cout << setw(4) << i / 100 << "% [";
            for (int j = 0; j < 40; j++)
            {
                if (j > i / 250)cout << "-";
                else cout << ">";
            }
            cout << "]";
        }
    }
    cout<<" test data cin completed!\n";
    fclose(stdin);
    freopen("E:\\code\\c++\\AI\\Neural Network\\MNIST_data\\train_image.txt", "r", stdin);
    for (int i = 1; i <= 60000; i++){
        for (int j = 0; j < 28*28; j++){
            train_dataset.push_back(rd());
        }
        if(i%100==0){
            cout << "\r";
            cout << setw(4) << i / 600 << "% [";
            for (int j = 0; j < 40; j++){
                if (j > i / 1500)cout << "-";
                else cout << ">";
            }
            cout << "]";
        }
    }
    cout << " train data cin completed!\n";
    fclose(stdin);
    
}
double train_once(double learning_rate) {
    double* input_dataset = new double[28 * 28];
    int index = cnt++;
    input.zero_gradient();
    for (int i = 0; i < 28 * 28; i++)
        input_dataset[i] = train_dataset[index * 28 * 28 + i];
    input.forward_once(input_dataset);
    output.backward_once(&train_labelset[index]);
    input.learn(learning_rate);
    return output.loss_sum();
}
bool test_once() {
    double* input_dataset = new double[28 * 28];
    int index = cnt++;
    input.zero_gradient();
    for (int i = 0; i < 28 * 28; i++)
        input_dataset[i] = test_dataset[index * 28 * 28 + i];
    input.forward_once(input_dataset);
    if (cnt % 100 == 0)printf("%d %d\n", output.get_ans(), int(test_labelset[index]));
    return output.get_ans() == int(test_labelset[index]);
}
using namespace std;


int main() {
    originate();
    srand(time(0));
    if (train_dataset.size() == 0)return -1;
    input.connect(&x1);
    x1.connect(&relu);
    relu.connect(&x3);
    x3.connect(&output);
    input.random_originate(-1, 1);
    const int T = 60000;
    cnt = 0;
    double loss_accu = 0;
    double changing_lr = 0.01;
    for (int epch = 0; epch < 3; epch++){
        cnt = 0;
        for (int i = 0; i < T; i++) {
            loss_accu += train_once(changing_lr);
            if ((i + 1) % (T / 100 > 0 ? T / 100 : 1) == 0) {
                printf("%d / %d: loss = %f\n", i + 1, T, loss_accu
                    / (T / 100 > 0 ? T / 100 : 1));
                loss_accu = 0;
            }
        }
    }
    const int test_T = 1000;
    int right = 0;
    cnt = 0;
    for (int i = 0; i < test_T; i++){
        if (test_once())right++;
    }
    printf("accuracy: %f", right * 1.0 / test_T);
    return 0;
}