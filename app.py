import streamlit as st
import os
import re
import fitz  # PyMuPDF
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
# Configuration
VLLM_BASE_URL = os.getenv("VLLM_API_BASE", "http://notebook-nlb-f68504b00671f364.elb.ap-northeast-2.amazonaws.com:8503/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen3-vl-32b-thinking")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
PDF_DATA_DIR = "test_pdf_data"

# Default Prompts (Matches CLI)
SYSTEM_PROMPT_DEFAULT = """당신은 기업의 **모든 사내 업무 매뉴얼(Internal Business Manuals)**을 텍스트 데이터베이스로 구축하는 **전문 테크니컬 라이터(Technical Writer)**입니다.
주어지는 이미지는 인사 규정,  IT 가이드, 재무 보고서, 안전 수칙, 운영 절차서(SOP) 등 다양한 사내 문서의 한 페이지입니다.

당신의 목표는 이미지 내의 정보를 시각적 요소 없이 오직 **'업무적 의미'와 '실질적 내용'**에 집중하여 구조화된 Markdown 문서로 완벽하게 변환하는 것입니다.

다음의 **[작성 원칙]**을 엄격히 준수하십시오:

### 1. 시각적 묘사 배제 (Context Over Visuals)
- **절대 금지:** 색상, 배치, 아이콘 모양, 장식적 이미지 등 디자인 요소에 대한 묘사는 하지 마십시오.
- **수행 지침:** 해당 이미지가 업무 수행을 위해 전달하고자 하는 **핵심 메시지, 규정, 데이터**만을 텍스트로 서술하십시오.

### 2. 비정형 데이터의 논리적 변환
- **UI/스크린샷 (시스템 화면):** 단순한 화면 묘사가 아닌, 사용자가 따라야 할 **'작업 절차(Actionable Steps)'**로 변환하십시오. (예: "저장 아이콘" -> "1. [저장] 버튼을 클릭하여 변경 사항을 반영합니다.")
- **도식/다이어그램 (구조 및 관계):** 조직도, 구성도, 네트워크 맵 등의 시각적 관계를 **계층형 리스트(Bulleted List)**나 **논리적 서술**로 풀어내십시오.
- **흐름도 (프로세스):** 업무 흐름이나 결재 라인 등의 화살표 흐름을 **'순서(Step 1, 2...)'** 또는 **'조건(If-Then)'** 문장으로 명확히 명시하십시오.

### 3. 표(Table) 데이터 처리
- **데이터 표:** 규정 수치, 스펙(Spec), 요율표, 일정 등 정확한 값이 중요한 표는 반드시 **Markdown Table** 문법을 사용하여 원본의 구조를 유지하십시오.
- **레이아웃용 표:** 단순히 배치를 위해 사용된 표는 텍스트의 흐름에 맞게 문장이나 리스트로 풀어서 작성하십시오.

### 4. 문서 구조화 (Formatting)
- 문서의 위계(장, 절, 항)를 파악하여 적절한 **Markdown Header (#, ##, ###)**를 적용하십시오.
- 본문 내용은 명확한 문단으로 구분하여 가독성을 높이십시오."""

USER_PROMPT_DEFAULT = """제공된 매뉴얼 페이지를 분석하여 DB 적재를 위한 **완벽한 Markdown 포맷**으로 출력해 주세요.

**[필수 수행 과제]**
1. **완전한 텍스트 추출:** 페이지 내의 모든 업무 관련 텍스트(본문, 주석, 캡션 포함)를 누락 없이 전사하십시오.
2. **구조적 명시:** 제목(#)과 본문, 리스트(-)를 명확히 구분하여 작성하십시오.
3. **불필요한 말 생략:** "분석 결과입니다"와 같은 서두나 맺음말 없이, 오직 **Markdown 본문 내용**만 출력하십시오."""



# Page Configuration
st.set_page_config(
    page_title="PDF Preprocessing with Qwen-VL",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #333; }
    .stButton > button { border-radius: 8px; }
    .reportview-container { margin-top: -2em; }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

def get_base64_image(pix):
    """Convert PyMuPDF binary data to base64 string"""
    data = pix.tobytes("jpg", jpg_quality=90)
    return base64.b64encode(data).decode('utf-8')

def parse_model_output(text):
    """
    Parse LLM output to separate <think>...</think> blocks from the actual response.
    Returns (thinking_content, response_content).
    """
    if not text:
        return "", ""
    
    # Check for thinking end tag FIRST (as per CLI robust fix)
    if '</think>' in text:
        parts = text.split('</think>')
        thinking = parts[0].replace('<think>', '').strip()
        response = parts[-1].strip()
        return thinking, response
        
    # Fallback to regex
    thinking_match = re.search(r'<think>(.*?)</think>', text, flags=re.IGNORECASE | re.DOTALL)
    thinking = thinking_match.group(1).strip() if thinking_match else ""
    
    response = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    
    return thinking, response

def process_page_with_qwen(system_prompt, user_prompt, base64_image, previous_context=""):
    """
    Returns a stream object from OpenAI client.
    """
    client = OpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
    )
    
    # Initialize list
    final_user_prompt_content = []

    # --- DEBUG LOGGING ---
    print("\n" + "="*50)
    print(f"🚀 Processing Page (Context: {'Yes' if previous_context else 'No'})")
    print(f"📷 Image Size: {len(base64_image)} chars")
    
    # 1. Add Context (Previous Page)
    if previous_context:
        print(f"🔗 Context Injected ({len(previous_context)} chars)")
        context_block = f"**[이전 페이지 내용 (문맥 유지용)]**\n{previous_context[-2000:]}\n\n**[지시사항]**\n위 문맥을 참고하여, 다음 페이지의 내용을 이어지는 형태로 자연스럽게 작성하시오."
        final_user_prompt_content.append({"type": "text", "text": context_block})
        
    # 2. Add Main Prompt & Image
    print(f"📝 User Prompt: {user_prompt[:100]}...")
    final_user_prompt_content.append({"type": "text", "text": user_prompt})
    final_user_prompt_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
        },
    })

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": final_user_prompt_content,
        }
    ]

    # Extra body for thinking mode (if needed by model/vllm)
    extra_body = {
        "chat_template_kwargs": {"enable_thinking": True}
    }

    try:
        print("⏳ Sending request to VLLM (Stream Mode)...")
        # Ensure stream=True
        stream = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            extra_body=extra_body,
            stream=True
        )
        return stream
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def list_pdf_files():
    if not os.path.exists(PDF_DATA_DIR):
        os.makedirs(PDF_DATA_DIR)
        return []
    return [f for f in os.listdir(PDF_DATA_DIR) if f.lower().endswith('.pdf')]

def render_pdf_page(pdf_path, page_num, target_long_side=1280):
    """
    Render a page to a target pixel size (long side).
    Ensures doc is closed to prevent memory leaks.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_num)
        
        # Calculate zoom to match target_long_side
        page_w, page_h = page.rect.width, page.rect.height
        max_dim = max(page_w, page_h)
        
        # Avoid dividing by zero and ensure target size
        if max_dim > 0:
            zoom = target_long_side / max_dim
        else:
            zoom = 1.0
            
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return pix
    finally:
        doc.close()

def main():
    st.title("👁️ Visual PDF Preprocessor")
    st.markdown("Convert PDF pages to rich text (context + visual descriptions) using Qwen-VL.")

    # Sidebar Settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # File Selection
        pdf_files = list_pdf_files()
        uploaded_file = st.file_uploader("Upload New PDF", type="pdf")
        
        selected_pdf_name = st.selectbox(
            "Select PDF from 'test_pdf_data'", 
            options=["-- Select --"] + pdf_files
        )

        current_pdf_path = None
        if uploaded_file:
            # Save uploaded file to temp path or data dir
            save_path = os.path.join(PDF_DATA_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            current_pdf_path = save_path
            st.success(f"Saved {uploaded_file.name}")
            # Refresh list workaround or just use this path
        elif selected_pdf_name != "-- Select --":
            current_pdf_path = os.path.join(PDF_DATA_DIR, selected_pdf_name)

        st.divider()
        
        if st.button("🚀 Process Entire Document", type="primary", use_container_width=True):
             st.session_state['batch_processing'] = True
             st.session_state['combined_result'] = "" # Reset

    if not current_pdf_path:
        st.info("👈 Please select or upload a PDF to start.")
        return

    # --- Main Content Area ---
    
    # Prompt Configuration (Prominent)
    st.header("📝 Prompt Configuration")
    st.markdown("These prompts control how the VLM interprets and translates your PDF pages.")
    
    col_sys, col_user = st.columns(2)
    with col_sys:
        system_prompt = st.text_area(
            "System Prompt",
            value=SYSTEM_PROMPT_DEFAULT,
            height=450
        )
    with col_user:
        user_prompt = st.text_area(
            "User Prompt",
            value=USER_PROMPT_DEFAULT,
            height=450
        )
    
    st.divider()

    # --- Document Processing ---
    doc = fitz.open(current_pdf_path)
    total_pages = len(doc)
    doc.close()

    # Check if batch processing was triggered
    if st.session_state.get('batch_processing', False):
        st.subheader("📚 Batch Processing")
        progress_bar = st.progress(0)
        status_text = st.empty()
        combined_text = ""
        
        last_page_text = ""
        
        for i in range(total_pages):
            status_text.text(f"Processing Page {i+1}/{total_pages}...")
            
            pix = render_pdf_page(current_pdf_path, i)
            base64_img = get_base64_image(pix)
            
            # Create a container for this page's result
            with st.container():
                st.markdown(f"### Page {i+1}")
                cols = st.columns([1, 2])
                
                # Column 1: Thumbnail
                with cols[0]:
                    st.image(pix.tobytes("jpg", jpg_quality=90), caption=f"Page {i+1}", use_container_width=True)
                
                # Column 2: Processing...
                with cols[1]:
                    # Create UI elements for streaming
                    thinking_expander = st.expander("🧠 Thinking Process", expanded=True)
                    thinking_placeholder = thinking_expander.empty()
                    
                    st.markdown("**📄 Extracted Content:**")
                    response_placeholder = st.empty()
                    
                    # Prepare for streaming
                    full_thinking = ""
                    full_response = ""
                    current_mode = "thinking" # thinking | response
                    
                    # Start Stream
                    stream = process_page_with_qwen(system_prompt, user_prompt, base64_img, previous_context=last_page_text)
                    
                    if stream:
                        for chunk in stream:
                            # Safely get content delta
                            if chunk.choices and chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                
                                # Check for transition: </think>
                                if "</think>" in content:
                                    parts = content.split("</think>")
                                    
                                    # First part goes to thinking
                                    full_thinking += parts[0]
                                    thinking_placeholder.code(full_thinking.replace("<think>", "").strip(), language='text')
                                    
                                    # Switch mode
                                    current_mode = "response"
                                    
                                    # Second part goes to response
                                    if len(parts) > 1:
                                        full_response += parts[1]
                                        response_placeholder.markdown(full_response + "▌")
                                        
                                    # Collapse thinking after done
                                    # thinking_expander.update(expanded=False) # Not directly possible in streamlit loop easily without rerun, ignore
                                    
                                else:
                                    if current_mode == "thinking":
                                        full_thinking += content
                                        # Only update thinking occasionally or it flickers? code block handles it well
                                        thinking_placeholder.code(full_thinking.replace("<think>", "").strip() + "▌", language='text')
                                    else:
                                        full_response += content
                                        response_placeholder.markdown(full_response + "▌")
                                        
                        # Final update to remove cursor
                        if current_thinking := full_thinking.replace("<think>", "").strip():
                            thinking_placeholder.code(current_thinking, language='text')
                        response_placeholder.markdown(full_response)
                        
                        # Set page result for next loop/storage
                        page_result = full_response
                        thinking = full_thinking
                    else:
                        st.error("Failed to connect to model.")
                        page_result = ""
            
            st.divider()
            
            combined_text += f"\n\n--- Page {i+1} ---\n\n"
            combined_text += page_result
            
            # Update last_page_text for next iteration context
            last_page_text = page_result
            
            progress_bar.progress((i + 1) / total_pages)
        
        st.session_state['combined_result'] = combined_text
        st.session_state['batch_processing'] = False
        status_text.success("Batch processing complete!")
        st.rerun()

    # Display Results
    if 'combined_result' in st.session_state and st.session_state['combined_result']:
        st.subheader("📑 Combined Document Output")
        st.text_area("Full Document Markdown", value=st.session_state['combined_result'], height=800)
        
        if st.button("💾 Save All to .txt"):
            output_filename = f"processed_{os.path.basename(current_pdf_path).replace('.pdf', '')}_FULL.txt"
            with open(os.path.join(PDF_DATA_DIR, output_filename), "w") as f:
                f.write(st.session_state['combined_result'])
            st.success(f"Saved to {output_filename}")
            
        if st.button(" Clear Results"):
             del st.session_state['combined_result']
             st.rerun()
    else:
        # Show preview when no results yet
        st.subheader(f"📄 Document Preview ({total_pages} pages)")
        page_num = st.slider("Preview Page", min_value=1, max_value=total_pages, value=1) - 1
        pix = render_pdf_page(current_pdf_path, page_num)
        st.image(pix.tobytes("jpg", jpg_quality=90), caption=f"Page {page_num + 1} / {total_pages}", use_container_width=True)

if __name__ == "__main__":
    main()