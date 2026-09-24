//! 组件目录（catalog）与预置候选（candidates）。
//!
//! 对应 json-render 的两层守卫：
//!
//! - **catalog**：允许出现的组件类型（渲染器只认这些 `kind`）。
//! - **candidates**：App 预先配置好的原子元素实例。Laya 只能在这些候选之间做
//!   选择，不能凭空生成组件、文案或数据——这正是 Jev 的硬约束。
//!
//! 文案、指标数值、图表数据、表格行全部都写死在这里，属于“App 提供的设计系统与
//! 内容”，模型只负责挑选与排序。

use serde_json::json;

use crate::spec::Element;

/// 一个允许的组件定义。
pub struct ComponentDef {
    pub name: &'static str,
    pub description: &'static str,
}

/// catalog：渲染器支持的组件类型集合。
pub const CATALOG: &[ComponentDef] = &[
    ComponentDef {
        name: "Page",
        description: "Root container with a title and one or more named columns (slots)",
    },
    ComponentDef {
        name: "Heading",
        description: "Section heading text",
    },
    ComponentDef {
        name: "Text",
        description: "Paragraph of explanatory text",
    },
    ComponentDef {
        name: "Metric",
        description: "Labeled numeric value",
    },
    ComponentDef {
        name: "Field",
        description: "Labeled field showing a value",
    },
    ComponentDef {
        name: "Toggle",
        description: "Labeled on/off switch",
    },
    ComponentDef {
        name: "BarChart",
        description: "Bar chart over labeled numeric data",
    },
    ComponentDef {
        name: "Table",
        description: "Table with columns and rows",
    },
    ComponentDef {
        name: "Badge",
        description: "Short status label",
    },
    ComponentDef {
        name: "Button",
        description: "Clickable action button",
    },
    ComponentDef {
        name: "Divider",
        description: "Horizontal separator",
    },
];

/// 一个候选元素：对某个组件的一次具体配置。
#[derive(Clone)]
pub struct Candidate {
    pub id: &'static str,
    /// 展示给决策模型的描述（决定它是否/如何被选中）。
    pub description: &'static str,
    /// 是否可作为页面根。
    pub root: bool,
    /// 互斥变体分组：同一 resource 的候选最多选一个。
    pub resource: Option<&'static str>,
    /// 组件的具体实例（type + props + 槽位）。
    pub element: Element,
}

/// 一个预置场景，打包了标题、请求提示与全部候选。
pub struct Preset {
    pub name: &'static str,
    pub title: &'static str,
    /// 供输入框与 `--print` 使用的示例请求。
    pub request_hint: &'static str,
    pub candidates: Vec<Candidate>,
}

/// 按名字返回预置场景。
pub fn preset(name: &str) -> Option<Preset> {
    match name {
        "dashboard" => Some(dashboard()),
        "settings" => Some(settings()),
        _ => None,
    }
}

/// 所有预置场景的名字。
pub fn names() -> &'static [&'static str] {
    &["dashboard", "settings"]
}

/// 销售 Dashboard：指标 + 柱状图 + 表格。
fn dashboard() -> Preset {
    Preset {
        name: "dashboard",
        title: "Sales Dashboard",
        request_hint: "Create a sales dashboard with a revenue chart and key metrics",
        candidates: vec![
            Candidate {
                id: "page_stack",
                description: "Single-column page that stacks sections vertically",
                root: true,
                resource: Some("page"),
                element: Element::new("Page")
                    .prop("title", json!("Sales Dashboard"))
                    .slot("main"),
            },
            Candidate {
                id: "page_split",
                description: "Two-column page with a primary column and a side column",
                root: true,
                resource: Some("page"),
                element: Element::new("Page")
                    .prop("title", json!("Sales Dashboard"))
                    .slot("main")
                    .slot("aside"),
            },
            Candidate {
                id: "heading",
                description: "Section heading: Weekly Sales Report",
                root: false,
                resource: None,
                element: Element::new("Heading").prop("text", json!("Weekly Sales Report")),
            },
            Candidate {
                id: "revenue_metric",
                description: "Key metric: Revenue $48.2K, up 12% week over week",
                root: false,
                resource: None,
                element: Element::new("Metric")
                    .prop("label", json!("Revenue"))
                    .prop("value", json!("$48.2K"))
                    .prop("delta", json!("+12%")),
            },
            Candidate {
                id: "orders_metric",
                description: "Key metric: Orders 1,284",
                root: false,
                resource: None,
                element: Element::new("Metric")
                    .prop("label", json!("Orders"))
                    .prop("value", json!("1,284"))
                    .prop("delta", json!("+4%")),
            },
            Candidate {
                id: "customers_metric",
                description: "Key metric: New customers 312",
                root: false,
                resource: None,
                element: Element::new("Metric")
                    .prop("label", json!("New customers"))
                    .prop("value", json!("312")),
            },
            Candidate {
                id: "revenue_chart",
                description: "Bar chart of revenue for the last seven days",
                root: false,
                resource: None,
                element: Element::new("BarChart")
                    .prop("title", json!("Revenue by day"))
                    .prop(
                        "data",
                        json!([
                            ["Mon", 42],
                            ["Tue", 55],
                            ["Wed", 48],
                            ["Thu", 63],
                            ["Fri", 71],
                            ["Sat", 38],
                            ["Sun", 52]
                        ]),
                    ),
            },
            Candidate {
                id: "orders_table",
                description: "Table listing the five most recent orders",
                root: false,
                resource: None,
                element: Element::new("Table")
                    .prop("title", json!("Recent orders"))
                    .prop("columns", json!(["Order", "Customer", "Total"]))
                    .prop(
                        "rows",
                        json!([
                            ["#1042", "Acme Co", "$1,200"],
                            ["#1041", "Globex", "$340"],
                            ["#1040", "Initech", "$89"],
                            ["#1039", "Umbrella", "$2,150"],
                            ["#1038", "Soylent", "$560"]
                        ]),
                    ),
            },
            Candidate {
                id: "live_badge",
                description: "Status badge: Live",
                root: false,
                resource: None,
                element: Element::new("Badge").prop("text", json!("LIVE")),
            },
            Candidate {
                id: "refresh_button",
                description: "Action button: Refresh the data",
                root: false,
                resource: None,
                element: Element::new("Button").prop("label", json!("Refresh")),
            },
            Candidate {
                id: "divider",
                description: "Horizontal divider separating sections",
                root: false,
                resource: None,
                element: Element::new("Divider"),
            },
            Candidate {
                id: "source_note",
                description: "Footnote describing the data source",
                root: false,
                resource: None,
                element: Element::new("Text")
                    .prop("text", json!("Source: offline dataset, updated just now.")),
            },
        ],
    }
}

/// 账户设置表单：字段 + 开关 + 按钮。
fn settings() -> Preset {
    Preset {
        name: "settings",
        title: "Account Settings",
        request_hint: "Compose an account settings page with profile fields and a save button",
        candidates: vec![
            Candidate {
                id: "settings_page",
                description: "Settings page that stacks fields vertically",
                root: true,
                resource: Some("page"),
                element: Element::new("Page")
                    .prop("title", json!("Account Settings"))
                    .slot("main"),
            },
            Candidate {
                id: "settings_split",
                description: "Settings page with fields on the left and a secondary column",
                root: true,
                resource: Some("page"),
                element: Element::new("Page")
                    .prop("title", json!("Account Settings"))
                    .slot("main")
                    .slot("aside"),
            },
            Candidate {
                id: "settings_heading",
                description: "Heading: Account settings",
                root: false,
                resource: None,
                element: Element::new("Heading").prop("text", json!("Account settings")),
            },
            Candidate {
                id: "name_field",
                description: "Editable field: Name is Chris Tate",
                root: false,
                resource: None,
                element: Element::new("Field")
                    .prop("label", json!("Name"))
                    .prop("value", json!("Chris Tate")),
            },
            Candidate {
                id: "email_field",
                description: "Editable field: Email is chris@example.com",
                root: false,
                resource: None,
                element: Element::new("Field")
                    .prop("label", json!("Email"))
                    .prop("value", json!("chris@example.com")),
            },
            Candidate {
                id: "role_field",
                description: "Read-only field: Role is Administrator",
                root: false,
                resource: None,
                element: Element::new("Field")
                    .prop("label", json!("Role"))
                    .prop("value", json!("Administrator")),
            },
            Candidate {
                id: "location_field",
                description: "Editable field: Location is San Francisco",
                root: false,
                resource: None,
                element: Element::new("Field")
                    .prop("label", json!("Location"))
                    .prop("value", json!("San Francisco")),
            },
            Candidate {
                id: "notifications_toggle",
                description: "Toggle: Email notifications, currently on",
                root: false,
                resource: None,
                element: Element::new("Toggle")
                    .prop("label", json!("Email notifications"))
                    .prop("on", json!(true)),
            },
            Candidate {
                id: "marketing_toggle",
                description: "Toggle: Product updates, currently off",
                root: false,
                resource: None,
                element: Element::new("Toggle")
                    .prop("label", json!("Product updates"))
                    .prop("on", json!(false)),
            },
            Candidate {
                id: "membership_badge",
                description: "Badge: Pro member",
                root: false,
                resource: None,
                element: Element::new("Badge").prop("text", json!("PRO")),
            },
            Candidate {
                id: "save_button",
                description: "Action button: Save changes",
                root: false,
                resource: None,
                element: Element::new("Button").prop("label", json!("Save changes")),
            },
            Candidate {
                id: "reset_button",
                description: "Action button: Reset the form",
                root: false,
                resource: None,
                element: Element::new("Button").prop("label", json!("Reset")),
            },
            Candidate {
                id: "settings_divider",
                description: "Horizontal divider separating sections",
                root: false,
                resource: None,
                element: Element::new("Divider"),
            },
        ],
    }
}
