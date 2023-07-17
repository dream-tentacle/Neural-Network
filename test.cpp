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
#include "CNN.cpp"
#include "Sigmoid.cpp"
const int train_size = 60000;
const int test_size = 10000;
std::vector<double> train_dataset,train_labelset,test_dataset,test_labelset;
Input_layer input(28 * 28);
Convolutional_neural_network cnn1(28, 28, 1, 3, 3, 8, 1);
Max_pool pool1(26, 26, 8, 2);
ReLU relu1(13 * 13 * 8);
Convolutional_neural_network cnn2(13, 13, 8, 3, 3, 16, 1);
Max_pool pool2(11, 11, 16, 2);
ReLU relu2(5 * 5 * 16);
Linear linear(5 * 5 * 16, 10);
Linear l2(13 * 13 * 16, 10);
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
    for (int i = 0; i < train_size; i++){
        train_labelset.push_back(rd());
    }
    fclose(stdin);
    freopen("E:\\code\\c++\\AI\\Neural Network\\MNIST_data\\test_label.txt", "r", stdin);
    for (int i = 0; i < test_size; i++){
        test_labelset.push_back(rd());
    }
    fclose(stdin);
    freopen("E:\\code\\c++\\AI\\Neural Network\\MNIST_data\\test_image.txt", "r", stdin);
    for (int i = 1; i <= test_size; i++)
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
    for (int i = 1; i <= train_size; i++){
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
    freopen("CON", "r", stdin);
}
double train_once(double learning_rate) {
    double* input_dataset = new double[28 * 28];
    int index = cnt++;
    index %= train_size;
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
    index %= test_size;
    input.zero_gradient();
    for (int i = 0; i < 28 * 28; i++)
        input_dataset[i] = test_dataset[index * 28 * 28 + i];
    input.forward_once(input_dataset);
    if (cnt % 1000 == 0)
        std::cout << output.get_ans() << " " << test_labelset[index]<<std::endl;
    return output.get_ans() == int(test_labelset[index]);
}
using namespace std;
void load_animation(int t) {
    cout << "\rloading ";
    if (t % 2 == 0)cout << "-";
    else if (t % 4 == 1)cout << "/";
    else cout << "\\";
    for (int i = 1; i <= 20; i++) {
        if (i % 3 == t % 3)cout << ">";
        else cout << " ";
    }
    cout << " ";
    if (t % 2 == 0)cout << "-";
    else if (t % 4 == 1)cout << "/";
    else cout << "\\";
}
int main() {
    originate();
    srand(time(0));
    if (train_dataset.size() == 0)return -1;
    input.connect(&cnn1);
    cnn1.connect(&pool1);
    pool1.connect(&relu1);
    relu1.connect(&cnn2);
    cnn2.connect(&pool2);
    pool2.connect(&relu2);
    relu2.connect(&linear);
    linear.connect(&output);
    input.random_originate(-0.1, 0.1);
    const int T = train_size;
    double loss_accu = 0;
    double changing_lr = 0.01;
    char ctnue = 'y';
    char show_loss = 'y';
    cout << "show loss£¿y/n ";
    while (ctnue == 'y') {
        if (show_loss == 'y') {
            cnt = 0;
            for (int i = 0; i < T; i++) {
                loss_accu += train_once(changing_lr);
                if ((i + 1) % (T / 10 > 0 ? T / 10 : 1) == 0) {
                    printf("%d / %d: loss = %f\n", i + 1, T, loss_accu
                        / (T / 10 > 0 ? T / 10 : 1));
                    loss_accu = 0;
                    //cnn1.print_gradient();
                    //puts("");
                    //cnn2.print_gradient();
                    //puts("");
                    //l2.print_gradient();
                    //puts("");
                }
            }
        }
        else {
            cnt = 0;
            for (int i = 0; i < T; i++) {
                loss_accu += train_once(changing_lr);
                load_animation(i / 100);
            }
        }
        const int test_T = 10000;
        int right = 0;
        cnt = 0;
        for (int i = 0; i < test_T; i++) {
            if (test_once())right++;
        }
        cout << "\r                                ";
        cout << "\raccuracy: " << right * 1.0 / test_T << endl;
        cout<<"continue training£¿y/n ";
        cin >> ctnue;
    }
    
    return 0;
}