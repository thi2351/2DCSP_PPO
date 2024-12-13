Vì môi trường không hiện thực đủ thông tin để code, nên tụi mình phải tự simulate observation ra.

# Các Module đã có sẵn:


## Product Encoder: 

Input: 1 tensor 2D shape (num_product, 3) (3 là width, height, quantity)
Output: 1 tensor 2D shape (num_product, 32)
Usage: Trích xuất features từ các attribute của nó (số 32 t set up)

## Stock Encoder:

Input: 1 tensor 3D shape (num_stock, max_w, max_h) 
Output : 1 tensor 3D shape (num_stock, 256)
Usage: Trích xuất features từ hình dạng của stock. Stock Encoder có sẵn preprocess lọc các vị trí đặt được là 1, các vị trí không đặt được là 0.

## DQN

Input: 1 tensor 1D shape (observation_dim,) là concatenated của stock_encoded và product_encoded
Output: 1 tensor 1D shape (num_of_possible_action, ) là Q-value của (a | s).
Usage: Maping từ observation sang action để quyết định chọn lựa. Index của action sẽ quyết định bộ (stock_idx, product_idx, rotated). Sau khi chọn được, ta tiến hành cắt heuristic như trong code.



Input: 1 tensor 1D shape (action_dim, ) là dim của bộ concatenated (stock_encoded[index1] <-> product_encoded[index2])
Output: 1 tensor 1D shape (1,) là V_estimate của Critic


## PPO

Kế thừa từ class Policy (do ông thầy bắt) và tùy thuộc vào policy_id mà sẽ init ra policy PPO hay policy cho việc cắt thường. 

Module này còn thiếu sót rất nhiều và chưa qua kiểm chứng...

## Reward Computation



# Các Module cần hiện thực:

## Masking

Input: 1 tensor 1D shape (num_of_possible_action, ) 
Output: 1 tensor 1D shape (num_of_possible_action, )
Usage: Để mask các giá trị được đánh dấu là invalid trong khi evaluate, tránh quá trình cắt không hợp lệ diễn ra lần 2 trong quá trình evaluate (train thì sai thoải mái, ăn phạt thôi)

## Observation initilization

Input: Observation
Output: 
   self.stocks: tensor (num_stock, max_w, max_h) 
   self.products: tensor (num_product, max_w, max_h)
Usage:
    Dùng để simulate môi trường lại và đọc thông tin thực hiện nội bộ trong agent.

## Observation update

Input: Action
Output:
    Update self.stocks và self.products dễ dàng dựa vào index của stock và product lấy ra từ action.

## Training 


## Save - Load model


## Evaluate model

