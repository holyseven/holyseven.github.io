document.addEventListener('DOMContentLoaded', function () {
  var filter = document.getElementById('tag-filter');
  if (!filter) return;

  var groups = {
    'llm-base': ['LLM', 'architecture', 'MoE', 'attention', 'long-context', 'pre-training', 'scaling', 'parameter-efficiency', 'latent-reasoning', 'adaptive-computation', 'transformer'],
    'training': ['RLHF', 'reinforcement-learning', 'alignment', 'AI-safety', 'Constitutional-AI', 'constitutional-AI', 'RLAIF', 'SFT', 'fine-tuning', 'post-training', 'data-diversity', 'data-efficiency', 'exploration', 'harmlessness', 'sycophancy', 'inverse-scaling', 'LLM-alignment', 'reward-model', 'token-level'],
    'eval': ['evaluation', 'benchmark', 'LLM-as-judge', 'LLM-as-Judge', 'LLM-evaluation', 'selective-prediction', 'calibration', 'conformal', 'experiments'],
    'agent': ['agent', 'LLM-agent', 'coding-agent', 'data-agent', 'multi-agent', 'agentic-coding', 'agent-benchmark', 'agent-design', 'harness-design', 'long-running-agent', 'tool-use', 'scheduling', 'resource-management', 'workplace-automation', 'GAN-inspired', 'evaluator', 'software-engineering'],
    'generation': ['text-generation', 'story-generation', 'consistency', 'creativity', 'creative-writing', 'divergent-thinking', 'personalization', 'preference-learning', 'essay-writing', 'literary-computing', 'narrative-analysis', 'conversation', 'multi-turn', 'reliability', 'NLP'],
    'code': ['code-generation', 'multilingual', 'Chinese-NLP', 'tokenization', 'BPE', 'robustness'],
    'diffusion': ['diffusion-model', 'flow-matching', 'language-model'],
    'systems': ['systems', 'distributed-training', 'parallelism', 'continued-pretraining', 'Megatron', 'Swift', 'enterprise-AI', 'RAG', 'OpenAI'],
    'bio': ['RNA', 'bioinformatics', 'pretrained-language-model', 'masked-language-model']
  };

  // Detect page type: index (post-list) or tags page (tag sections)
  var postList = document.getElementById('post-list');

  if (postList) {
    // Index page: filter <li> items
    var posts = postList.querySelectorAll('li');
    filter.addEventListener('click', function (e) {
      var btn = e.target.closest('.tag-btn');
      if (!btn) return;
      var group = btn.getAttribute('data-tag');
      setActive(btn);
      posts.forEach(function (li) {
        li.style.display = matchGroup(li.getAttribute('data-tags'), group) ? '' : 'none';
      });
    });
  } else {
    // Tags page: filter <h3> + <ul> sections
    var sections = document.querySelectorAll('h3[id^="tag-"]');
    filter.addEventListener('click', function (e) {
      var btn = e.target.closest('.tag-btn');
      if (!btn) return;
      var group = btn.getAttribute('data-tag');
      setActive(btn);
      sections.forEach(function (h3) {
        var tagName = h3.id.replace('tag-', '');
        var ul = h3.nextElementSibling;
        var show = group === 'all' || (groups[group] && groups[group].indexOf(tagName) !== -1);
        h3.style.display = show ? '' : 'none';
        if (ul) ul.style.display = show ? '' : 'none';
      });
    });
  }

  function setActive(btn) {
    filter.querySelectorAll('.tag-btn').forEach(function (b) {
      b.classList.remove('active');
    });
    btn.classList.add('active');
  }

  function matchGroup(tagsStr, group) {
    if (group === 'all') return true;
    var postTags = tagsStr.split(',');
    var groupTags = groups[group] || [];
    return postTags.some(function (t) {
      return groupTags.indexOf(t) !== -1;
    });
  }
});
