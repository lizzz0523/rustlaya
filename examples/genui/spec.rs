//! 与 `json-render` 对齐的 flat Spec 数据模型。
//!
//! Spec 是 `{ root, elements }`：一棵由字符串 id 寻址的扁平组件树。组件只能
//! 来自 catalog（守卫），树结构由 composer 依据 Laya 的离散决策组装。字段名刻意
//! 保持 `type` / `props` / `children` / `slots`，便于与 json-render 对照。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 组件 props：有序的 JSON 对象（`serde_json` 开启了 `preserve_order`）。
pub type Props = serde_json::Map<String, Value>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Element {
    /// 组件类型（catalog 中的名字）。
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Props::is_empty")]
    pub props: Props,
    /// 默认槽位之外的无名子节点。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<String>,
    /// 命名槽位 -> 子节点 id 列表（对应 json-render 的 named slots）。
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub slots: IndexMap<String, Vec<String>>,
}

impl Element {
    pub fn new(kind: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            props: Props::new(),
            children: Vec::new(),
            slots: IndexMap::new(),
        }
    }

    /// 链式设置一个 prop。
    pub fn prop(mut self, key: &str, value: Value) -> Self {
        self.props.insert(key.to_string(), value);
        self
    }

    /// 声明一个命名槽位（先声明后填充，保持顺序）。
    pub fn slot(mut self, name: &str) -> Self {
        self.slots.entry(name.to_string()).or_default();
        self
    }

    /// 清空结构，仅保留 kind 与 props（用于把候选模板实例化为新节点）。
    pub fn as_leaf(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            props: self.props.clone(),
            children: Vec::new(),
            slots: IndexMap::new(),
        }
    }

    /// 克隆为容器：保留 kind/props 与已声明的槽位键，但清空子节点及各槽内容。
    pub fn as_container(&self) -> Self {
        let mut element = self.clone();
        element.children.clear();
        for children in element.slots.values_mut() {
            children.clear();
        }
        element
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Spec {
    /// 根元素 id；空树为 `None`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    /// id -> 元素。
    #[serde(default)]
    pub elements: IndexMap<String, Element>,
}

impl Spec {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: impl Into<String>, element: Element) -> String {
        let id = id.into();
        self.elements.insert(id.clone(), element);
        id
    }

    pub fn get(&self, id: &str) -> Option<&Element> {
        self.elements.get(id)
    }

    /// 把 `id` 挂到 `parent` 的命名槽（`Some`）或无名子节点（`None`）末尾。
    pub fn attach(&mut self, id: &str, parent: &str, slot: Option<&str>) {
        let Some(element) = self.elements.get_mut(parent) else {
            return;
        };
        match slot {
            Some(name) => element
                .slots
                .entry(name.to_string())
                .or_default()
                .push(id.to_string()),
            None => element.children.push(id.to_string()),
        }
    }

    /// 从树中摘除 `id` 及其整个子树，并清理所有父节点引用。
    pub fn detach(&mut self, id: &str) {
        let mut stack = vec![id.to_string()];
        let mut removed: Vec<String> = Vec::new();
        while let Some(current) = stack.pop() {
            if let Some(element) = self.elements.shift_remove(&current) {
                stack.extend(element.children.iter().cloned());
                for children in element.slots.values() {
                    stack.extend(children.iter().cloned());
                }
                removed.push(current);
            }
        }
        for element in self.elements.values_mut() {
            element.children.retain(|child| !removed.contains(child));
            for children in element.slots.values_mut() {
                children.retain(|child| !removed.contains(child));
            }
        }
    }

    /// 把 `id` 移到其所在父级列表 / 槽位的第一位。
    pub fn move_to_front(&mut self, id: &str) {
        for element in self.elements.values_mut() {
            if let Some(position) = element.children.iter().position(|child| child == id) {
                let item = element.children.remove(position);
                element.children.insert(0, item);
                return;
            }
            for children in element.slots.values_mut() {
                if let Some(position) = children.iter().position(|child| child == id) {
                    let item = children.remove(position);
                    children.insert(0, item);
                    return;
                }
            }
        }
    }

    /// 供编辑 prompt 使用的、当前页面已有元素的简短描述列表。
    pub fn summary(&self) -> Vec<String> {
        self.elements
            .iter()
            .filter(|(id, _)| Some(id.as_str()) != self.root.as_deref())
            .map(|(id, element)| format!("{id} ({})", element.kind))
            .collect()
    }
}
