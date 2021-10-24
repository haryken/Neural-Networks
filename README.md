# Neural Networks

Mạng nơ-ron nhân tạo (Neural Network - NN) là một mô hình lập trình rất đẹp lấy cảm hứng từ mạng nơ-ron thần kinh. Kết hợp với các kĩ thuật học sâu (Deep Learning - DL), NN đang trở thành một công cụ rất mạnh mẽ mang lại hiệu quả tốt nhất cho nhiều bài toán khó như nhận dạng ảnh, giọng nói hay xử lý ngôn ngữ tự nhiên.

## 1. Perceptrons
### 1.1. Perceptron cơ bản
Một mạng nơ-ron được cấu thành bởi các nơ-ron đơn lẻ được gọi là các perceptron. Nên trước tiên ta tìm hiểu xem perceptron là gì đã rồi tiến tới mô hình của mạng nơ-ron sau. Nơ-ron nhân tạo được lấy cảm hứng từ nơ-ron sinh học như hình mô tả bên dưới:

![image](https://user-images.githubusercontent.com/64195026/138585440-0d48223b-3f58-4b5b-9579-7def4b85db01.png)

Như hình trên, ta có thể thấy một nơ-ron có thể nhận nhiều đầu vào và cho ra một kết quả duy nhất. Mô hình của perceptron cũng tương tự như vậy:

![image](https://user-images.githubusercontent.com/64195026/138585451-ba43e6fe-d3c9-4d4e-b0a9-9f9788d6ccde.png)

Một perceptron sẽ nhận một hoặc nhiều đầu \mathbf{x}x vào dạng nhị phân và cho ra một kết quả oo dạng nhị phân duy nhất. Các đầu vào được điều phối tầm ảnh hưởng bởi các tham số trọng lượng tương ứng \mathbf{w}w của nó, còn kết quả đầu ra được quyết định dựa vào một ngưỡng quyết định bb nào đó:

![image](https://user-images.githubusercontent.com/64195026/138585467-e4ffc3f6-965b-4c0c-a311-ee461ea573b5.png)

Đặt b=-threshold, ta có thể viết lại thành:

![image](https://user-images.githubusercontent.com/64195026/138585500-fdc6f839-3a81-4eba-a9d6-bafd7dd88d54.png)

Để dễ hình dung, ta lấy ví dụ việc đi nhậu hay không phụ thuộc vào 4 yếu tố sau:

+ 1. Trời có nắng hay không?
+ 2. Có hẹn trước hay không?
+ 3. Vợ có vui hay không?
+ 4. Bạn nhậu có ít khi gặp được hay không?

Thì ta coi 4 yếu tố đầu vào là x1, x2, x3, x4 và nếu o=0 thì ta không đi nhậu còn o=1o=1 thì ta đi nhậu. Giả sử mức độ quan trọng của 4 yếu tố trên lần lượt là w_1=0.05, w_2=0.5, w_3=0.2, w_4=0.25 và chọn ngưỡng b=-0.5 thì ta có thể thấy rằng việc trời nắng có ảnh hưởng chỉ 5% tới quyết định đi nhậu và việc có hẹn từ trước ảnh hưởng tới 50% quyết định đi nhậu của ta.
 
 Nếu gắn x0=1 và w_0=b, ta còn có thể viết gọn lại thành:
 
 ![image](https://user-images.githubusercontent.com/64195026/138585619-a844af9f-31c1-475a-8e8d-3447e0d6db1c.png)

### 1.2. Sigmoid Neurons

Với đầu vào và đầu ra dạng nhị phân, ta rất khó có thể điều chỉnh một lượng nhỏ đầu vào để đầu ra thay đổi chút ít, nên để linh động, ta có thể mở rộng chúng ra cả khoảng [0,1]. Lúc này đầu ra được quyết định bởi một hàm sigmoid ![image](https://user-images.githubusercontent.com/64195026/138585656-b1b0e5c9-18a9-4c6d-b454-3bd075968503.png). Như các bài trước đã đề cập thì hàm sigmoid có công thức:

![image](https://user-images.githubusercontent.com/64195026/138585690-10cf649f-d0f9-41ed-bda9-aed16c386208.png)

Đồ thị của hàm này cũng cân xứng rất đẹp thể hiện được mức độ công bằng của các tham số:

![image](https://user-images.githubusercontent.com/64195026/138585702-99f14cb2-f0f5-4607-a845-24a185be593a.png)

Đặt ![image](https://user-images.githubusercontent.com/64195026/138585719-26a517aa-7e1a-45ec-8b31-a0145b4e426a.png) thì công thức của perceptron lúc này sẽ có dạng:

![image](https://user-images.githubusercontent.com/64195026/138585728-a2362b01-3d13-43d5-9333-2eee23e76295.png)

Tới đây thì ta có thể thấy rằng mỗi sigmoid neuron cũng tương tự như một bộ phân loại tuyến tính (logistic regression) bởi xác suất ![image](https://user-images.githubusercontent.com/64195026/138585737-85d9b6d7-85ed-43bc-94fe-a1e461d4bcd0.png)

hực ra thì ngoài hàm sigmoid ra, ta còn có thể một số hàm khác như tanhtanh,ReLU để thay thế hàm sigmoid bởi dạng đồ thị của nó cũng tương tự như sigmoid. Một cách tổng quát, hàm perceptron được biểu diễn qua một hàm kích hoạt (activation function) f(z) như sau:
 
![image](https://user-images.githubusercontent.com/64195026/138585753-4d80d18d-ef9d-45b0-989b-00b3ffe27b57.png)

Bằng cách biểu diễn như vậy, ta có thể coi neuron sinh học được thể hiện như sau:

![image](https://user-images.githubusercontent.com/64195026/138585765-7184019b-f87a-4b4a-b26a-511ad1404d9d.png)

Một điểm cần lưu ý là các hàm kích hoạt buộc phải là hàm phi tuyến. Vì nếu nó là tuyến tính thì khi kết hợp với phép toán tuyến tính ![image](https://user-images.githubusercontent.com/64195026/138585776-8301ed6c-41a7-46dd-86eb-7e99f6613290.png) thì kết quả thu được cũng sẽ là một thao tác tuyến tính dẫn tới chuyện nó trở nên vô nghĩa.

## 2. Kiến trúc mạng NN
Mạng NN là sự kết hợp của của các tầng perceptron hay còn được gọi là perceptron đa tầng (multilayer perceptron) như hình vẽ bên dưới:

![image](https://user-images.githubusercontent.com/64195026/138585783-caa30a9d-a9ce-4bcd-9d58-5042a3a3d8c7.png)

Một mạng NN sẽ có 3 kiểu tầng:

+ Tầng vào (input layer): Là tầng bên trái cùng của mạng thể hiện cho các đầu vào của mạng.
+ Tầng ra (output layer): Là tầng bên phải cùng của mạng thể hiện cho các đầu ra của mạng.
+ Tầng ẩn (hidden layer): Là tầng nằm giữa tầng vào và tầng ra thể hiện cho việc suy luận logic của mạng.

Lưu ý rằng, một NN chỉ có 1 tầng vào và 1 tầng ra nhưng có thể có nhiều tầng ẩn.

![image](https://user-images.githubusercontent.com/64195026/138585794-ebd4aac2-ddcf-4e39-a0a4-12fc91162205.png)

Trong mạng NN, mỗi nút mạng là một sigmoid nơ-ron nhưng hàm kích hoạt của chúng có thể khác nhau. Tuy nhiên trong thực tế người ta thường để chúng cùng dạng với nhau để tính toán cho thuận lợi.

Ở mỗi tầng, số lượng các nút mạng (nơ-ron) có thể khác nhau tuỳ thuộc vào bài toán và cách giải quyết. Nhưng thường khi làm việc người ta để các tầng ẩn có số lượng nơ-ron bằng nhau. Ngoài ra, các nơ-ron ở các tầng thường được liên kết đôi một với nhau tạo thành mạng kết nối đầy đủ (full-connected network). Khi đó ta có thể tính được kích cỡ của mạng dựa vào số tầng và số nơ-ron. Ví dụ ở hình trên ta có:

![image](https://user-images.githubusercontent.com/64195026/138585801-d1168fe0-9515-413c-aeae-e1295f0e616b.png)

## 3. Lan truyền tiến
Như bạn thấy thì tất cả các nốt mạng (nơ-ron) được kết hợp đôi một với nhau theo một chiều duy nhất từ tầng vào tới tầng ra. Tức là mỗi nốt ở một tầng nào đó sẽ nhận đầu vào là tất cả các nốt ở tầng trước đó mà không suy luận ngược lại. Hay nói cách khác, việc suy luận trong mạng NN là suy luận tiến (feedforward):

![image](https://user-images.githubusercontent.com/64195026/138585806-f7fc507e-2478-4c94-9bef-ba7efee5033c.png)

## 4. Học với mạng NN
Cũng tương tự như các bài toán học máy khác thì quá trình học vẫn là tìm lấy một hàm lỗi để đánh giá và tìm cách tối ưu hàm lỗi đó để được kết quả hợp lý nhất có thể. Như đã đề cập mỗi nút mạng của NN có thể coi là một bộ phân loại (logistic regression) có hàm lỗi là:

![image](https://user-images.githubusercontent.com/64195026/138585830-27aaeb38-de88-45a0-868d-c65f88f3d812.png)

Trong đó, mm là số lượng dữ liệu huấn luyện, ![image](https://user-images.githubusercontent.com/64195026/138585842-a89dbd1b-cbaa-4230-b422-c9fb07adb00f.png) là đầu ra thực tế của dữ liệu thứ ii trong tập huấn luyện. Còn ![image](https://user-images.githubusercontent.com/64195026/138585853-38385fc5-2675-40ab-84b9-ea92e6deb612.png) là kết quả ước lượng được ứng với dữ liệu thứ ii.

Hàm lỗi của NN cũng tương tự như vậy, chỉ khác là đầu ra của mạng NN có thể có nhiều nút nên khi tính đầu ra ta cũng cần phải tính cho từng nút ra đó. Giả sử số nút ra là KK và ![image](https://user-images.githubusercontent.com/64195026/138585869-351c2a8c-226e-42eb-a90c-c363634315ae.png) là đầu ra thực tế của nút thứ kk, còn ![image](https://user-images.githubusercontent.com/64195026/138585881-dcdb2154-c5fe-4124-b963-963c63465d1d.png) là đầu ra ước lượng được cho nút thứ kk tương ứng. Khi đó, công thức tính hàm lỗi sẽ thành:

![image](https://user-images.githubusercontent.com/64195026/138585888-aae4b95e-57cc-4f61-ae4f-7ad6ef24a8e3.png)

Lưu ý rằng, các tham số lúc này không còn đơn thuần là một ma trận nữa mà là một tập của tất cả các ma trận tham số của tất cả các tầng mạng nên tôi biểu diễn nó dưới dạng tập hợp W

Để tối ưu hàm lỗi ta vẫn sử dụng các phương pháp đạo hàm như đã đề cập ở các bài viết trước. Nhưng việc tính đạo hàm lúc này không đơn thuần như logistic regression bởi để ước lượng được đầu ra ta phải trải qua quá trình lan truyền tiến. Tức là để tính được ![image](https://user-images.githubusercontent.com/64195026/138585947-32ee18e6-a82a-434d-909d-7c1bc630a69c.png) ta cần một loạt các phép tính liên hợp nhau.

## 5. Lan truyền ngược và đạo hàm
Để tính đạo hàm của hàm lỗi ![image](https://user-images.githubusercontent.com/64195026/138585907-e4119069-a60f-4633-a930-e29501733f34.png) trong mạng NN, ta sử dụng một giải thuật đặc biệt là giải thuật lan truyền ngược (backpropagation). Nhờ có giải thuật được sáng tạo vào năm 1986 này mà mạng NN thực thi hiệu quả được và ứng dụng ngày một nhiều cho tới tận ngày này.

Về cơ bản phương pháp này được dựa theo quy tắc chuỗi đạo hàm của hàm hợp và phép tính ngược đạo hàm để thu được đạo hàm theo tất cả các tham số cùng lúc chỉ với 2 lần duyệt mạng. Tuy nhiên trong bài viết này, tôi chỉ đề cập ngay tới công thức tính toán còn việc chứng minh thì tôi sẽ dành cho các bài tiếp theo.

Giải thuật lan truyền ngược được thực hiện như sau:

+ 1. Lan truyền tiến:

![image](https://user-images.githubusercontent.com/64195026/138585966-d240df8d-7683-408d-a762-99cf6ca4224e.png)

+ 2. Tính đạo hàm theo z ở tầng ra:

![image](https://user-images.githubusercontent.com/64195026/138585984-4d7d11d2-8e9b-4dc0-becc-2db20fcf03f1.png)

+ 3. Lan truyền ngược:

![image](https://user-images.githubusercontent.com/64195026/138585992-f0668123-81aa-4129-ace0-744da141c577.png)


+ 4. Tính đạo hàm:

![image](https://user-images.githubusercontent.com/64195026/138585999-f0e26b77-ab37-4905-b523-a2a81a59420f.png)


