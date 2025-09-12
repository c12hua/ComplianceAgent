<template>
  <div class="chat-container">
    <div class="app-header">
      <div class="app-branding">
        <h1 class="app-title">PII检测</h1>
      </div>
      <div class="header-controls">
        <div class="model-select-wrapper">
          <n-select
            class="model-select"
            v-model:value="selectedModel"
            :options="modelOptions"
            placeholder="选择检测模型"
            size="large"
          />
        </div>
        <div class="header-actions">
          <n-button class="header-button" quaternary circle size="large">
            <template #icon>
              <n-icon size="24"><UserOutlined /></n-icon>
            </template>
          </n-button>
          <n-button class="header-button" quaternary circle size="large">
            <template #icon>
              <n-icon size="24"><SettingOutlined /></n-icon>
            </template>
          </n-button>
          <n-button class="theme-toggle" circle quaternary size="large">
            <template #icon>
              <n-icon size="24">
                <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              </n-icon>
            </template>
          </n-button>
        </div>
      </div>
    </div>

    <!-- Messages Area -->
    <div class="chat-messages" ref="messagesRef">
      <div v-if="piiResult" class="message">
        <div class="message-content">
          <div class="pii-result" :class="{ 'is-typing': isTyping }">
            <!-- 原始文本部分 -->
            <div v-if="piiResult.text" class="result-section text-section">
              <div class="section-header">
                <h3>原始文本</h3>
              </div>
              <p class="text-content">{{ displayedText }}</p>
            </div>

            <!-- 风险概览部分 -->
            <div v-if="piiResult?.summary" class="result-section summary-section">
              <div class="risk-overview">
                <div class="risk-summary-header">
                  <div class="risk-summary-main">
                    <div class="risk-badge" :class="displayedSummary.risk_level">
                      <span class="risk-icon">{{ 
                        displayedSummary.risk_level === '高' ? '🔴' : 
                        displayedSummary.risk_level === '中' ? '🟡' : '🟢' 
                      }}</span>
                      <span class="risk-level">{{ displayedSummary.risk_level }}风险</span>
                      <span class="entity-count">发现 {{ displayedSummary.total_entities }} 项敏感信息</span>
                    </div>
                    <div class="selected-model">
                      <span class="model-icon">🤖</span>
                      <span class="model-name">{{ selectedModel === 'deepseek' ? 'DeepSeek 智能分析' : 
                        selectedModel === 'presidio' ? 'Presidio 规则识别' : 
                        selectedModel === 'gpt4' ? 'GPT-4 高级识别' :
                        selectedModel === 'claude' ? 'Claude 智能分析' : '智能识别' }}</span>
                    </div>
                  </div>
                </div>
                <div class="risk-explanation">
                  <div class="explanation-header">
                    <h4>风险分析摘要</h4>
                    <span class="law-reference">根据《个人信息保护法》与《医疗健康信息安全指南》</span>
                  </div>
                  <div class="explanation-content">
                    <p>{{ displayedSummary.overall_reason }}</p>
                  </div>
                </div>
              </div>
            </div>

            <!-- 详细信息部分 -->
            <div v-if="piiResult.details?.length" class="result-section details-section">
              <div class="section-header">
                <div class="section-title">
                  <h3>敏感信息详情</h3>
                  <span class="total-details-badge">共{{ piiResult.details.length }}项</span>
                </div>
                <p class="section-description">根据相关法律法规，以下信息需要重点保护</p>
              </div>
              <div class="details-grid">
       <div v-for="(detail, index) in displayedDetails" 
         :key="index" 
         class="detail-card"
         :class="detail.risk_level">
                  <div class="detail-header">
                    <div class="detail-header-main">
                      <div class="risk-tag" :class="detail.risk_level">
                        <span class="risk-icon">{{ detail.risk_level === '高' ? '🔴' : detail.risk_level === '中' ? '🟡' : '🟢' }}</span>
                        <span>{{ detail.risk_level }}风险</span>
                      </div>
                      <div class="risk-type-tag">
                        {{ 
                          detail.entities[0].includes('身份证') || detail.entities[0].includes('护照') ? '身份证件' :
                          detail.entities[0].includes('电话') || detail.entities[0].includes('手机') || detail.entities[0].includes('邮箱') ? '联系方式' :
                          detail.entities[0].includes('地址') ? '地理位置' :
                          detail.entities[0].includes('过敏') || detail.entities[0].includes('诊断') || detail.entities[0].includes('病历') ? '健康信息' :
                          detail.entities[0].includes('银行') || detail.entities[0].includes('账号') ? '金融信息' :
                          detail.entities[0].includes('车牌') ? '车辆信息' :
                          detail.entities[0].includes('姓名') ? '个人身份' :
                          detail.entities[0].includes('公司') || detail.entities[0].includes('单位') ? '组织信息' :
                          detail.entities[0].includes('日期') || detail.entities[0].includes('生日') ? '时间信息' :
                          detail.entities[0].includes('照片') || detail.entities[0].includes('图像') ? '生物特征' :
                          detail.entities[0].includes('社交') ? '社交关系' :
                          '敏感信息'
                        }}
                      </div>
                    </div>
                    <div class="entity-count-badge">{{ detail.entities.length > 1 ? `${detail.entities.length}项信息` : '单项信息' }}</div>
                  </div>
                  <div class="detail-content">
                    <div class="entities-section">
                      <div class="section-title-bar">
                        <h4>检测到的敏感信息</h4>
                        <span class="info-tag">需要保护</span>
                      </div>
                      <div class="entities-list">
                        <div v-for="(entity, entityIndex) in detail.entities" 
                             :key="entityIndex"
                             class="entity-item"
                             :class="detail.risk_level">
                          <span class="entity-icon">{{ 
                            entity.includes('身份证') ? '🆔' :
                            entity.includes('电话') || entity.includes('手机') ? '📱' :
                            entity.includes('地址') ? '📍' :
                            entity.includes('过敏') || entity.includes('诊断') || entity.includes('病历') ? '🏥' :
                            entity.includes('邮箱') ? '📧' :
                            entity.includes('银行') || entity.includes('账号') ? '💳' :
                            entity.includes('护照') ? '🛂' :
                            entity.includes('车牌') ? '🚗' :
                            entity.includes('姓名') ? '👤' :
                            entity.includes('公司') || entity.includes('单位') ? '🏢' :
                            entity.includes('日期') || entity.includes('生日') ? '📅' :
                            entity.includes('照片') || entity.includes('图像') ? '🖼️' :
                            entity.includes('社交') ? '💬' :
                            '🔒'
                          }}</span>
                          <span class="entity-text">{{ entity }}</span>
                        </div>
                      </div>
                    </div>
                    <div class="analysis-section">
                      <div class="section-title-bar">
                        <h4>风险分析</h4>
                        <span class="law-tag">法律依据</span>
                      </div>
                      <div class="detail-reason">
                        <p>{{ detail.reason }}</p>
                      </div>
                    </div>
                    <!-- 向量检索命中 (可折叠) -->
                    <div v-if="detail.matches?.length" class="matches-section">
                      <div class="matches-header">
                        <h4>检索证据</h4>
                        <button class="toggle-matches" @click="toggleMatches(index)">
                          {{ detail._showMatches ? '收起' : `展开 ${detail.matches.length} 条证据` }}
                        </button>
                      </div>
                      <div v-if="detail._showMatches" class="matches-list">
                        <div v-for="(m, mi) in detail.matches.slice(0,2)" :key="mi" class="match-item">
                          <div class="match-snippet">{{ m.text_snippet || m.snippet || m.answer || m.related_text || '（无摘录）' }}</div>
                          <div class="match-meta">
                            <span class="match-source">来源: {{ getMatchSource(m) }}</span>
                            <span class="match-score">分数: {{ m.score }}</span>
                          </div>
                          <div class="match-related" v-if="m.related_answers">相关回答: {{ m.related_answers.join(' | ') }}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Description Area -->
    <div class="description-area" v-if="!piiResult">
      <div class="description-content">
        <p class="description-text">请输入任意文本，智能检测系统将自动识别并标注其中的敏感信息</p>
        <p class="description-subtext">支持识别姓名、身份证号、手机号、地址等多种敏感信息</p>
      </div>
    </div>

    <!-- Input Area -->
    <div class="chat-input-container">
      <div class="input-wrapper">
        <n-input
          v-model:value="inputText"
          type="textarea"
          :placeholder="placeholder"
          class="chat-input"
          :autosize="{ minRows: 1, maxRows: 6 }"
          @keydown.enter.prevent="handleEnter"
        />
        <div class="input-buttons">
          <button class="icon-button" :disabled="!inputText" @click="clearInput">
            <n-icon><CloseOutlined /></n-icon>
          </button>
          <button
            class="send-button"
            :disabled="!inputText || loading"
            :class="{ 'is-loading': loading }"
            @click="detectPII"
          >
            <n-icon v-if="!loading">
              <SendOutlined />
            </n-icon>
            <div v-else class="loading-spinner">
              <span class="dot"></span>
              <span class="dot"></span>
              <span class="dot"></span>
            </div>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, h } from 'vue'
import { NInput, NSelect, NButton, NIcon } from 'naive-ui'
import { UserOutlined, SettingOutlined, SendOutlined, CloseOutlined } from '@vicons/antd'
import { detectPII as apiDetectPII } from '@/api/pii'

const inputText = ref('')
const loading = ref(false)
const piiResult = ref(null)
const messagesRef = ref(null)
const selectedModel = ref('deepseek')

// 添加打字机效果相关的响应式变量
const displayedText = ref('')  // 用于显示原始文本
const displayedSummary = ref({  // 用于显示风险概览
  risk_level: '',
  total_entities: 0,
  overall_reason: ''
})
const displayedDetails = ref([])  // 用于显示详细信息 (每个 detail may include .matches)
const isTyping = ref(false)  // 是否正在打字
const currentStep = ref(0)  // 当前展示步骤

// 打字机效果的速度控制
const typeSpeed = 50  // 打字速度（毫秒/字符）
const stepDelay = 800  // 每个步骤之间的延迟（毫秒）

// 渲染模型选项
const modelOptions = [
  {
    label: 'DeepSeek (智能识别)',
    value: 'deepseek',
    description: '基于大模型的智能识别，适合复杂场景'
  },
  {
    label: 'GPT-4 (高级识别)',
    value: 'gpt4',
    description: '基于 GPT-4 的高级识别，支持多语言和复杂上下文'
  },
  {
    label: 'Claude (智能分析)',
    value: 'claude',
    description: '基于 Claude 的智能分析，擅长处理长文本和专业领域'
  },
  {
    label: 'Presidio (规则分析)',
    value: 'Presidio',
    description: '基于 Presidio的规则分析，擅长处理结构化的规则文件'
  },
  {
    label: 'LLaMA2 (轻量级)',
    value: 'llama2',
    description: '开源大模型，性能与效率的平衡选择'
  }
]

const placeholder = '请输入需要检测的文本...'

const clearInput = () => {
  inputText.value = ''
}

const handleEnter = (e) => {
  if (!e.shiftKey && !e.isComposing) {
    detectPII()
  }
}

// 实现打字效果的函数
const typeWriter = async (target, text, speed = typeSpeed) => {
  let index = 0
  return new Promise((resolve) => {
    const type = () => {
      if (index < text.length) {
        target.value += text[index]
        index++
        setTimeout(type, speed)
      } else {
        resolve()
      }
    }
    type()
  })
}

// 展示结果的动画序列
const animateResults = async (skipText = false) => {
  if (!piiResult.value) return
  
  isTyping.value = true
  currentStep.value = 0
  
  // 清空分析结果显示
  displayedSummary.value = { risk_level: '', total_entities: 0, overall_reason: '' }
  displayedDetails.value = []
  
  // 只在需要时清空并显示文本
  if (!skipText) {
    displayedText.value = ''
    // 步骤1: 显示原始文本
    currentStep.value = 1
    await typeWriter(displayedText, piiResult.value.text)
    await new Promise(resolve => setTimeout(resolve, stepDelay))
  }
  
  try {
    // 步骤2: 显示风险概览
    if (piiResult.value.summary) {
      currentStep.value = 2
      Object.assign(displayedSummary.value, {
        risk_level: piiResult.value.summary.risk_level,
        total_entities: piiResult.value.summary.total_entities,
        overall_reason: ''
      })
      await typeWriter(displayedSummary.value, piiResult.value.summary.overall_reason, typeSpeed * 1.5)
      displayedSummary.value.overall_reason = piiResult.value.summary.overall_reason
      await new Promise(resolve => setTimeout(resolve, stepDelay))
    }
    
    // 步骤3: 逐个显示详细信息
    if (piiResult.value.details?.length) {
      currentStep.value = 3
        displayedDetails.value = []  // 清空现有详情
      
      for (const detail of piiResult.value.details) {
        // 创建新的详情对象
          const newDetail = {
            risk_level: detail.risk_level,
            entities: [],
            reason: '',
            // matches 是后端增加的向量检索证据数组 (可选)
            matches: detail.matches || []
          }
        displayedDetails.value.push(newDetail)
        
        // 显示实体
        for (const entity of detail.entities) {
          await new Promise(resolve => setTimeout(resolve, typeSpeed * 3))
          newDetail.entities.push(entity)
        }
        
  // 设置原因文本
  newDetail.reason = detail.reason
        await new Promise(resolve => setTimeout(resolve, stepDelay))
      }
    }
  } finally {
    isTyping.value = false
  }
}

const detectPII = async () => {
  if (!inputText.value || loading.value) return

  loading.value = true
  // 立即显示原始文本
  const originalText = inputText.value
  inputText.value = ''
  
  try {
    // 立即设置并显示原始文本
    piiResult.value = {
      text: originalText,
      summary: null,
      details: []
    }
    displayedText.value = originalText
    
    // 滚动到底部以显示原始文本
    setTimeout(() => {
      if (messagesRef.value) {
        messagesRef.value.scrollTop = messagesRef.value.scrollHeight
      }
    }, 100)
    
    // 发送请求并等待响应
    const response = await apiDetectPII(originalText, selectedModel.value)
    // 后端返回的数据可能包含独立的 fields:
    // - entities: [{entity, type}, ...]
    // - entity_assessments: { '<entity>': {risk, reason}, ... }
    // - entity_vector_matches: { '<entity>': [ {score, text_snippet, metadata, related_answers}, ... ] }
    const data = response.data || {}

    // 构建合并后的 details 数组，兼容后端直接返回 details 的场景
    let details = []
    if (Array.isArray(data.details) && data.details.length) {
      // 如果后端已经返回 details，尽量使用并补充 matches/assessment
      details = data.details.map(d => ({
        risk_level: d.risk_level || (data.entity_assessments && data.entity_assessments[d.entities?.[0]]?.risk) || '',
        entities: d.entities || [],
        reason: d.reason || (data.entity_assessments && data.entity_assessments[d.entities?.[0]]?.reason) || '',
        matches: d.matches || (data.entity_vector_matches && data.entity_vector_matches[d.entities?.[0]]) || []
      }))
    } else if (Array.isArray(data.entities) && data.entities.length) {
      // 从 entities 构建每个 detail
      details = data.entities.map(e => {
        const entText = typeof e === 'string' ? e : (e.entity || '')
        return {
          risk_level: (data.entity_assessments && data.entity_assessments[entText]?.risk) || '',
          entities: [entText],
          reason: (data.entity_assessments && data.entity_assessments[entText]?.reason) || '',
          matches: (data.entity_vector_matches && data.entity_vector_matches[entText]) || []
        }
      })
    }

  // 将重构后的结构放回 piiResult 以供后续动画与渲染使用
  piiResult.value = Object.assign({}, data, { details })

  // 初始化显示用的 flags 并把 details 赋给 displayedDetails
  displayedDetails.value = initDetailFlags(details)

  // 开始动画展示分析结果（跳过原始文本显示）
  await animateResults(true)
    
  } catch (error) {
    console.error('PII detection failed:', error)
  } finally {
    loading.value = false
  }
}

// Toggle showing matches for a detail card
const toggleMatches = (index) => {
  const d = displayedDetails.value[index]
  if (!d) return
  d._showMatches = !d._showMatches
}

// Ensure any details created have _showMatches default false
// Watcher-like helper used when assigning displayedDetails
const initDetailFlags = (details) => {
  for (const d of details) {
    if (d._showMatches === undefined) d._showMatches = false
    if (!Array.isArray(d.matches)) d.matches = []
    // cap to top-2 matches for display
    if (d.matches.length > 2) d.matches = d.matches.slice(0, 2)
  }
  return details
}

// Extract a readable source/filename from a match metadata object
const getMatchSource = (m) => {
  if (!m) return '未知来源'
  const md = m.metadata || m.meta || m.source || {}
  if (typeof md === 'string') return md
  // common fields: filename, source, doc, path
  return md.filename || md.file || md.source || md.doc || md.path || '未知来源'
}
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
  padding: 2rem 0;
}

.app-header {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem 2.5rem;
  background: rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--gpt-border);
  z-index: 10;
}

.app-branding {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.app-title {
  font-size: 2rem;
  font-weight: 700;
  color: var(--gpt-text);
  margin: 0;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}

.header-controls {
  display: flex;
  align-items: center;
}

.model-select-wrapper {
  margin-right: 16px;
  width: 240px;
}

.model-option {
  padding: 8px 12px;
  cursor: pointer;
}

.model-option-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.model-option-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--gpt-text);
}

.model-option-desc {
  font-size: 12px;
  color: var(--gpt-text-secondary);
  opacity: 0.8;
  line-height: 1.4;
}

:deep(.n-select-menu) {
  max-width: 240px !important;
}

:deep(.n-base-selection) {
  background: var(--n-color) !important;
  min-width: 240px !important;
}

:deep(.n-base-selection-label) {
  font-size: 14px !important;
}

:deep(.n-base-selection-placeholder) {
  font-size: 14px !important;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
}

.header-button,
.theme-toggle {
  width: 44px !important;
  height: 44px !important;
  border-radius: 12px !important;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--gpt-text);
  transition: all 0.3s ease;
  background: rgba(255, 255, 255, 0.8) !important;
  border: 1px solid var(--gpt-border) !important;
}

.header-button:hover,
.theme-toggle:hover {
  transform: translateY(-2px);
  background: rgba(255, 255, 255, 0.9) !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  margin-top: 4rem;
}

.message {
  display: flex;
  justify-content: center;
  padding: 1.5rem 1rem;
}

.message-content {
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem;
}

.pii-result {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.result-section {
  background: var(--gpt-content-bg);
  border-radius: 1rem;
  margin-bottom: 2rem;
  border: 1px solid var(--gpt-border);
  overflow: hidden;
}

.section-header {
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--gpt-border);
  background: rgba(0, 0, 0, 0.02);
}

.section-header h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--gpt-text);
  margin: 0;
}

.result-section p,
.result-section ul {
  margin: 0;
  line-height: 1.6;
}

.result-section ul {
  padding-left: 1.5rem;
}

/* 风险概览样式 */
.risk-overview {
  padding: 1.5rem;
}

.risk-summary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.risk-badge {
  display: inline-flex;
  align-items: center;
  padding: 0.75rem 1.5rem;
  border-radius: 1rem;
  font-weight: 600;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.risk-badge.高 {
  background-color: #fef2f2;
  color: #dc2626;
  border: 1px solid rgba(220, 38, 38, 0.2);
}

.risk-badge.中 {
  background-color: #fff7ed;
  color: #ea580c;
  border: 1px solid rgba(234, 88, 12, 0.2);
}

.risk-badge.低 {
  background-color: #f0fdf4;
  color: #16a34a;
  border: 1px solid rgba(22, 163, 74, 0.2);
}

.risk-level {
  font-size: 1.25rem;
  margin-right: 1rem;
  font-weight: 700;
}

.entity-count {
  font-size: 1rem;
  opacity: 0.9;
  padding-left: 1rem;
  border-left: 2px solid currentColor;
}

.selected-model {
  display: flex;
  align-items: center;
  padding: 0.5rem 1rem;
  background: rgba(0, 0, 0, 0.03);
  border-radius: 0.75rem;
  font-size: 0.9rem;
}

.model-icon {
  margin-right: 0.5rem;
}

.risk-explanation {
  background: rgba(0, 0, 0, 0.02);
  padding: 1.5rem;
  border-radius: 1rem;
  margin-top: 1rem;
}

.risk-explanation h4 {
  margin: 0 0 1rem 0;
  font-size: 1.1rem;
  color: #374151;
}

.risk-explanation p {
  font-size: 1.1rem;
  line-height: 1.6;
  color: #4b5563;
  margin: 0;
}

/* 详细信息网格布局 */
.details-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem;
  padding: 1.5rem;
}

.section-header {
  padding: 1.5rem 2rem;
}

.section-header h3 {
  margin: 0;
  font-size: 1.5rem;
  color: #111827;
}

.section-description {
  margin: 0.5rem 0 0;
  color: #6b7280;
}

.detail-card {
  background: var(--gpt-content-bg);
  border-radius: 1rem;
  border: 1px solid rgba(0, 0, 0, 0.1);
  overflow: hidden;
  transition: all 0.3s ease;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.detail-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.detail-card.高 {
  border-top: 4px solid #dc2626;
}

.detail-card.中 {
  border-top: 4px solid #ea580c;
}

.detail-card.低 {
  border-top: 4px solid #16a34a;
}

.detail-header {
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  background: rgba(0, 0, 0, 0.02);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.risk-tag {
  display: flex;
  align-items: center;
  padding: 0.5rem 1rem;
  border-radius: 0.75rem;
  font-size: 1rem;
  font-weight: 600;
}

.risk-icon {
  margin-right: 0.5rem;
}

.entity-count-badge {
  font-size: 0.875rem;
  color: #6b7280;
  background: rgba(0, 0, 0, 0.05);
  padding: 0.25rem 0.75rem;
  border-radius: 1rem;
}

.detail-content {
  padding: 1.5rem;
}

.entities-section,
.analysis-section {
  margin-bottom: 1.5rem;
}

.entities-section h4,
.analysis-section h4 {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: #374151;
}

.entities-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.entity-item {
  display: flex;
  align-items: center;
  font-size: 0.95rem;
  padding: 0.5rem 1rem;
  border-radius: 0.75rem;
  background: rgba(0, 0, 0, 0.03);
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.entity-icon {
  margin-right: 0.5rem;
}

.entity-text {
  color: #111827;
}

.detail-reason {
  font-size: 1rem;
  color: #4b5563;
  line-height: 1.6;
  background: rgba(0, 0, 0, 0.02);
  padding: 1rem;
  border-radius: 0.75rem;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.matches-section {
  margin-top: 1rem;
}
.matches-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}
.matches-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.match-item {
  padding: 0.75rem;
  border-radius: 0.6rem;
  background: rgba(0,0,0,0.03);
  border: 1px solid rgba(0,0,0,0.05);
}
.match-snippet {
  color: #111827;
  margin-bottom: 0.5rem;
}
.match-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: #6b7280;
}
.match-source { font-weight: 600; }
.match-score { color: #9ca3af }
.toggle-matches {
  background: transparent;
  border: none;
  color: #3b82f6;
  cursor: pointer;
}

/* 文本内容样式 */
.text-content {
  padding: 1.5rem;
  font-size: 1.1rem;
  line-height: 1.6;
  color: var(--gpt-text);
}

/* 打字机效果相关样式 */
.typing-cursor {
  display: inline-block;
  width: 2px;
  height: 1.2em;
  background-color: currentColor;
  margin-left: 2px;
  animation: blink 1s infinite;
  vertical-align: middle;
}

.typing-indicator {
  display: inline-flex;
  align-items: center;
  margin-left: 8px;
}

.typing-indicator::after {
  content: "...";
  animation: typing 1s infinite;
  font-weight: bold;
}

.animate-fade-in {
  animation: fadeIn 0.5s ease-out;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

@keyframes typing {
  0%, 100% { content: "."; }
  33% { content: ".."; }
  66% { content: "..."; }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 动画效果 */
.message-enter-active,
.message-leave-active {
  transition: all 0.3s ease;
}

.message-enter-from,
.message-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

@media (max-width: 768px) {
  .details-grid {
    grid-template-columns: 1fr;
  }
  
  .message-content {
    padding: 1rem;
  }
}

.chat-input-container {
  position: relative;
  padding: 1.5rem 1rem;
  border-top: 1px solid var(--gpt-border);
}

.input-wrapper {
  position: relative;
  max-width: min(65vw, 60rem);
  margin: 0 auto;
  background: var(--gpt-content-bg);
  border: 1px solid var(--gpt-border);
  border-radius: 1.5rem;
  padding: 1rem;
  box-shadow: 0 0 15px rgba(0,0,0,0.1);
  transition: all 0.3s ease;
}

.input-wrapper:hover {
  box-shadow: 0 0 20px rgba(0,0,0,0.15);
}

.chat-input {
  width: 100%;
  padding-right: 6rem;
}

:deep(.n-input) {
  background: transparent !important;
}

:deep(.n-input__textarea) {
  min-height: 4.5rem !important;
  padding: 1rem !important;
  border-radius: 1.25rem !important;
  font-size: 1rem !important;
  line-height: 1.5 !important;
  background: transparent !important;
}

.input-buttons {
  position: absolute;
  right: 1rem;
  bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.icon-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border: none;
  background: var(--gpt-button-bg);
  color: var(--gpt-text);
  cursor: pointer;
  border-radius: 0.75rem;
  transition: all 0.3s ease;
  border: 1px solid var(--gpt-border);
}

.icon-button:hover {
  background: var(--gpt-hover-bg);
  transform: scale(1.05);
}

.icon-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.send-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border: none;
  background: var(--gpt-text);
  color: var(--gpt-bg);
  cursor: pointer;
  border-radius: 0.75rem;
  transition: all 0.3s ease;
}

.send-button:hover:not(:disabled) {
  transform: scale(1.05);
  opacity: 0.9;
}

.send-button:disabled {
  background: var(--gpt-border);
  cursor: not-allowed;
  transform: none;
}

.send-button.is-loading {
  background: var(--gpt-text);
  opacity: 0.8;
  cursor: wait;
}

.loading-spinner {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
}

.loading-spinner .dot {
  width: 4px;
  height: 4px;
  background-color: var(--gpt-bg);
  border-radius: 50%;
  display: inline-block;
  animation: bounce 1.4s infinite ease-in-out both;
}

.loading-spinner .dot:nth-child(1) {
  animation-delay: -0.32s;
}

.loading-spinner .dot:nth-child(2) {
  animation-delay: -0.16s;
}

@keyframes bounce {
  0%, 80%, 100% { 
    transform: scale(0);
  } 
  40% { 
    transform: scale(1);
  }
}

/* 描述文本区域样式 */
.description-area {
  display: flex;
  justify-content: center;
  padding: 2rem 1rem;
  text-align: center;
}

.description-content {
  max-width: 800px;
}

.description-text {
  font-size: 1.5rem;
  font-weight: 500;
  color: var(--gpt-text);
  margin-bottom: 1rem;
  line-height: 1.4;
}

.description-subtext {
  font-size: 1.1rem;
  color: var(--gpt-text-secondary);
  opacity: 0.8;
}

/* 模型选择渲染相关样式 */
.model-option {
  padding: 4px 0;
}

.model-option-name {
  font-weight: 500;
  font-size: 1rem;
  margin-bottom: 2px;
}

.model-option-desc {
  font-size: 0.85rem;
  color: var(--gpt-text-secondary);
  opacity: 0.8;
}

.risk-summary-main {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  margin-bottom: 1rem;
}

.explanation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.law-reference {
  font-size: 0.875rem;
  color: #6b7280;
  padding: 0.25rem 0.75rem;
  background: rgba(59, 130, 246, 0.1);
  border-radius: 1rem;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.total-details-badge {
  padding: 0.25rem 0.75rem;
  background: rgba(59, 130, 246, 0.1);
  border-radius: 1rem;
  font-size: 0.875rem;
  color: #3b82f6;
}

.section-title-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.info-tag, .law-tag {
  font-size: 0.75rem;
  padding: 0.125rem 0.5rem;
  border-radius: 1rem;
  background: rgba(59, 130, 246, 0.1);
  color: #3b82f6;
}

.risk-type-tag {
  font-size: 0.875rem;
  padding: 0.25rem 0.75rem;
  border-radius: 1rem;
  background: rgba(0, 0, 0, 0.05);
  color: #4b5563;
}

.detail-header-main {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.entity-item.高 {
  border-left: 3px solid #dc2626;
}

.entity-item.中 {
  border-left: 3px solid #ea580c;
}

.entity-item.低 {
  border-left: 3px solid #16a34a;
}

.explanation-content {
  background: rgba(59, 130, 246, 0.05);
  padding: 1rem;
  border-radius: 0.75rem;
  margin-top: 0.5rem;
}

.explanation-content p {
  margin: 0;
  line-height: 1.6;
  color: #1f2937;
}
</style>
