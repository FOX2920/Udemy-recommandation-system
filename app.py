import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Đọc dữ liệu
df = pd.read_csv('Data/udemy_courses.csv')

# Tạo TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Tính toán ma trận TF-IDF từ mô tả khóa học
tfidf_matrix = tfidf.fit_transform(df['course_title'])

# Hàm đề xuất
def recommend_courses(course_idx, top_n=5):
    # Tính toán sự tương đồng giữa khóa học được chọn và tất cả các khóa học khác
    cosine_sim = cosine_similarity(tfidf_matrix[course_idx], tfidf_matrix)
    
    # Lấy chỉ số của các khóa học tương tự
    similar_indices = cosine_sim.argsort()[0][-top_n-1:-1][::-1]
    
    # Lấy thông tin khóa học tương ứng
    recommended_courses = df.iloc[similar_indices][['course_title', 'url']]
    
    return recommended_courses

# Giao diện Streamlit
st.title('Hệ thống đề xuất khóa học')

# Hiển thị danh sách các khóa học
st.subheader('Danh sách khóa học')
course_list = df['course_title'].tolist()
selected_course = st.selectbox('Chọn một khóa học', course_list)

# Hiển thị khóa học được chọn
st.write(f'Bạn đã chọn khóa học: {selected_course}')

# Đề xuất các khóa học tương tự
if st.button('Đề xuất khóa học'):
    course_idx = df[df['course_title'] == selected_course].index[0]
    recommended_courses = recommend_courses(course_idx)
    st.subheader('Các khóa học được đề xuất:')
    for i, row in recommended_courses.iterrows():
        st.write(f"- [{row['course_title']}]({row['url']})")
