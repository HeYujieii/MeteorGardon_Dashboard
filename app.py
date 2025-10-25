import os
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from pymongo import MongoClient

load_dotenv()

# Postgres schema helper
PG_SCHEMA = os.getenv("PG_SCHEMA", "public")
def qualify(sql: str) -> str:
    return sql.replace("{S}.", f"{PG_SCHEMA}.")

# CONFIG: Postgres and Mongo Queries
CONFIG = {
    "postgres": {
        "enabled": True,
        "uri": os.getenv("PG_URI", "postgresql+psycopg2://postgres:password@localhost:5432/postgres"),
        "queries": {
            # User 1: MEMBERS
            "Member: my workout sessions (table)": {
                "sql": """
                    SELECT session_id, session_date, duration, calories_consumed, intensity_level
                    FROM {S}.Workout_Session
                    WHERE member_id = :member_id
                    ORDER BY session_date DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["member"],
                "params": ["member_id"]
            },
            "Member: calories burned this week (bar)": {
                "sql": """
                    SELECT session_date::date as date, SUM(calories_consumed) as total_calories
                    FROM {S}.Workout_Session
                    WHERE member_id = :member_id
                      AND session_date >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY session_date::date
                    ORDER BY date;
                """,
                "chart": {"type": "bar", "x": "date", "y": "total_calories"},
                "tags": ["member"],
                "params": ["member_id"]
            },
            "Member: equipment usage summary (table)": {
                "sql": """
                    SELECT e.equipment_name, SUM(wse.usage_duration) as total_usage
                    FROM {S}.Workout_Session_Equipment wse
                    JOIN {S}.Equipment e ON wse.equipment_id = e.equipment_id
                    JOIN {S}.Workout_Session ws ON wse.session_id = ws.session_id
                    WHERE ws.member_id = :member_id
                    GROUP BY e.equipment_name
                    ORDER BY total_usage DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["member"],
                "params": ["member_id"]
            },
            "Member: workout intensity distribution (pie)": {
                "sql": """
                    SELECT intensity_level, COUNT(*) as session_count
                    FROM {S}.Workout_Session
                    WHERE member_id = :member_id
                    GROUP BY intensity_level;
                """,
                "chart": {"type": "pie", "names": "intensity_level", "values": "session_count"},
                "tags": ["member"],
                "params": ["member_id"]
            },

            # User 2: PERSONAL TRAINERS
            "Trainer: clients overview (table)": {
                "sql": """
                    SELECT m.member_id, m.age, m.gender, m.height, m.weight, m.credit_score
                    FROM {S}.Member m
                    WHERE m.trainer_id = :trainer_id
                    ORDER BY m.credit_score DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["trainer"],
                "params": ["trainer_id"]
            },
            "Trainer: client session completion rate (bar)": {
                "sql": """
                    SELECT m.member_id, 
                           COUNT(ws.session_id) as total_sessions,
                           ROUND(COUNT(ws.session_id) * 100.0 / NULLIF(:target_sessions, 0), 1) as completion_rate
                    FROM {S}.Member m
                    LEFT JOIN {S}.Workout_Session ws ON m.member_id = ws.member_id
                    WHERE m.trainer_id = :trainer_id
                      AND ws.session_date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY m.member_id
                    ORDER BY completion_rate DESC;
                """,
                "chart": {"type": "bar", "x": "member_id", "y": "completion_rate"},
                "tags": ["trainer"],
                "params": ["trainer_id", "target_sessions"]
            },
            "Trainer: workout plans (table)": {
                "sql": """
                    SELECT plan_id, plan_name, difficulty_level, cycle
                    FROM {S}.Workout_Plan
                    WHERE trainer_id = :trainer_id
                    ORDER BY difficulty_level;
                """,
                "chart": {"type": "table"},
                "tags": ["trainer"],
                "params": ["trainer_id"]
            },
            "Trainer: high intensity sessions (table)": {
                "sql": """
                    SELECT m.member_id, ws.session_date, ws.duration, ws.calories_consumed
                    FROM {S}.Workout_Session ws
                    JOIN {S}.Member m ON ws.member_id = m.member_id
                    WHERE m.trainer_id = :trainer_id
                      AND ws.intensity_level = 'High'
                    ORDER BY ws.session_date DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["trainer"],
                "params": ["trainer_id"]
            },

            # User 3: GYM MANAGERS
            "Manager: equipment usage frequency (bar)": {
                "sql": """
                    SELECT e.equipment_name, COUNT(wse.session_id) as usage_count
                    FROM {S}.Equipment e
                    LEFT JOIN {S}.Workout_Session_Equipment wse ON e.equipment_id = wse.equipment_id
                    LEFT JOIN {S}.Workout_Session ws ON wse.session_id = ws.session_id
                    WHERE ws.session_date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY e.equipment_name
                    ORDER BY usage_count DESC;
                """,
                "chart": {"type": "bar", "x": "equipment_name", "y": "usage_count"},
                "tags": ["manager"]
            },
            "Manager: maintenance schedule (table)": {
                "sql": """
                    SELECT equipment_name, type, maintenance_date
                    FROM {S}.Equipment
                    WHERE maintenance_date <= CURRENT_DATE + INTERVAL '30 days'
                    ORDER BY maintenance_date;
                """,
                "chart": {"type": "table"},
                "tags": ["manager"]
            },
            "Manager: member demographics (table)": {
                "sql": """
                    SELECT 
                        COUNT(*) as total_members,
                        AVG(age) as avg_age,
                        AVG(height) as avg_height,
                        AVG(weight) as avg_weight,
                        AVG(credit_score) as avg_credit_score
                    FROM {S}.Member;
                """,
                "chart": {"type": "table"},
                "tags": ["manager"]
            },
            "Manager: peak hour analysis (bar)": {
                "sql": """
                    SELECT EXTRACT(HOUR FROM session_date) as hour, COUNT(*) as session_count
                    FROM {S}.Workout_Session
                    WHERE session_date >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY EXTRACT(HOUR FROM session_date)
                    ORDER BY hour;
                """,
                "chart": {"type": "bar", "x": "hour", "y": "session_count"},
                "tags": ["manager"]
            },

            # User 4: SYSTEM ADMINISTRATORS
            "Admin: device inventory (table)": {
                "sql": """
                    SELECT d.device_id, m.member_id, d.model, d.brand, d.purchase_date
                    FROM {S}.Device d
                    LEFT JOIN {S}.Member m ON d.member_id = m.member_id
                    ORDER BY d.purchase_date DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["admin"]
            },
            "Admin: sensor calibration status (table)": {
                "sql": """
                    SELECT s.sensor_id, s.type, s.calibration_date, d.model as device_model
                    FROM {S}.Sensor s
                    JOIN {S}.Device d ON s.device_id = d.device_id
                    WHERE s.calibration_date <= CURRENT_DATE - INTERVAL '90 days'
                    ORDER BY s.calibration_date;
                """,
                "chart": {"type": "table"},
                "tags": ["admin"]
            },
            "Admin: data quality overview (table)": {
                "sql": """
                    SELECT data_quality, COUNT(*) as reading_count
                    FROM {S}.Sensor_Reading
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
                    GROUP BY data_quality
                    ORDER BY reading_count DESC;
                """,
                "chart": {"type": "table"},
                "tags": ["admin"]
            }
        }
    },

    "mongo": {
        "enabled": True,
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "db_name": os.getenv("MONGO_DB", "smart_fitness_ecosystem"),
        
        "queries": {
            "TS: Hourly avg heart rate (member M_1001, last 24h)": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$match": {
                        "member_id": "M_1001",
                        "type": "Heart Rate Monitor",
                        "timestamp": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}
                    }},
                    {"$project": {
                        "hour": {"$dateTrunc": {"date": "$timestamp", "unit": "hour"}},
                        "value": 1
                    }},
                    {"$group": {"_id": "$hour", "avg_hr": {"$avg": "$value"}, "n": {"$count": {}}}},
                    {"$sort": {"_id": 1}}
                ],
                "chart": {"type": "line", "x": "_id", "y": "avg_hr"}
            },

            "TS: High quality sensor readings (last 7 days)": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$match": {
                        "timestamp": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=7)},
                        "data_quality": "high"
                    }},
                    {"$group": {"_id": "$sensor_id", "high_quality_count": {"$count": {}}}},
                    {"$sort": {"high_quality_count": -1}}
                ],
                "chart": {"type": "bar", "x": "_id", "y": "high_quality_count"}
            },

            "Telemetry: Latest sensor readings per device": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$sort": {"timestamp": -1}},
                    {"$group": {"_id": "$sensor_id", "doc": {"$first": "$$ROOT"}}},
                    {"$replaceRoot": {"newRoot": "$doc"}},
                    {"$project": {
                        "_id": 0, "sensor_id": 1, "member_id": 1, "timestamp": 1,
                        "value": 1, "data_quality": 1, "type": 1
                    }}
                ],
                "chart": {"type": "table"}
            },

            "Analytics: Data quality distribution": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$match": {"timestamp": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=1)}}},
                    {"$group": {"_id": "$data_quality", "count": {"$count": {}}}},
                    {"$sort": {"count": -1}}
                ],
                "chart": {"type": "pie", "names": "_id", "values": "count"}
            },

            "TS Treemap: readings count by member and sensor type (last 24h)": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$match": {"timestamp": {"$gte": dt.datetime.utcnow() - dt.timedelta(hours=24)}}},
                    {"$group": {"_id": {"member": "$member_id", "sensor_type": "$type"}, "count": {"$count": {}}}},
                    {"$project": {"member": "$_id.member", "sensor_type": "$_id.sensor_type", "count": 1, "_id": 0}}
                ],
                "chart": {"type": "treemap", "path": ["member", "sensor_type"], "values": "count"}
            },

            "Analytics: Workout session sensor trends": {
                "collection": "sensor_readings",
                "aggregate": [
                    {"$match": {
                        "workout_session_id": {"$exists": True},
                        "timestamp": {"$gte": dt.datetime.utcnow() - dt.timedelta(days=7)}
                    }},
                    {"$group": {
                        "_id": {"session": "$workout_session_id", "type": "$type"},
                        "avg_value": {"$avg": "$value"},
                        "max_value": {"$max": "$value"},
                        "readings_count": {"$count": {}}
                    }},
                    {"$sort": {"readings_count": -1}},
                    {"$limit": 20}
                ],
                "chart": {"type": "table"}
            }
        }
    }
}

# Streamlit dashboard configuration
st.set_page_config(page_title="Smart Fitness Ecosystem Dashboard", layout="wide")
st.title("Smart Fitness Ecosystem | Dashboard (Postgres + MongoDB)")

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for (k, v), c in zip(metrics.items(), cols):
        c.metric(k, v)

@st.cache_resource
def get_pg_engine(uri: str):
    return create_engine(uri, pool_pre_ping=True, future=True)

@st.cache_data(ttl=60)
def run_pg_query(_engine, sql: str, params: dict | None = None):
    with _engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

@st.cache_resource
def get_mongo_client(uri: str):
    return MongoClient(uri)

def mongo_overview(client: MongoClient, db_name: str):
    info = client.server_info()
    db = client[db_name]
    colls = db.list_collection_names()
    stats = db.command("dbstats")
    total_docs = sum(db[c].estimated_document_count() for c in colls) if colls else 0
    return {
        "DB": db_name,
        "Collections": f"{len(colls):,}",
        "Total docs (est.)": f"{total_docs:,}",
        "Storage": f"{round(stats.get('storageSize',0)/1024/1024,1)} MB",
        "Version": info.get("version", "unknown")
    }

@st.cache_data(ttl=60)
def run_mongo_aggregate(_client, db_name: str, coll: str, stages: list):
    db = _client[db_name]
    docs = list(db[coll].aggregate(stages, allowDiskUse=True))
    return pd.json_normalize(docs) if docs else pd.DataFrame()

def render_chart(df: pd.DataFrame, spec: dict):
    if df.empty:
        st.info("No rows.")
        return
    ctype = spec.get("type", "table")
    # light datetime parsing for x axes
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass

    if ctype == "table":
        st.dataframe(df, use_container_width=True)
    elif ctype == "line":
        st.plotly_chart(px.line(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "bar":
        st.plotly_chart(px.bar(df, x=spec["x"], y=spec["y"]), use_container_width=True)
    elif ctype == "pie":
        st.plotly_chart(px.pie(df, names=spec["names"], values=spec["values"]), use_container_width=True)
    elif ctype == "heatmap":
        pivot = pd.pivot_table(df, index=spec["rows"], columns=spec["cols"], values=spec["values"], aggfunc="mean")
        st.plotly_chart(px.imshow(pivot, aspect="auto", origin="upper",
                                  labels=dict(x=spec["cols"], y=spec["rows"], color=spec["values"])),
                        use_container_width=True)
    elif ctype == "treemap":
        st.plotly_chart(px.treemap(df, path=spec["path"], values=spec["values"]), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

# Dashboard sidebar
with st.sidebar:
    st.header("Connections")
    pg_uri = st.text_input("Postgres URI", CONFIG["postgres"]["uri"])     
    mongo_uri = st.text_input("Mongo URI", CONFIG["mongo"]["uri"])        
    mongo_db = st.text_input("Mongo DB name", CONFIG["mongo"]["db_name"]) 
    st.divider()
    auto_run = st.checkbox("Auto-run on selection change", value=False, key="auto_run_global")

    st.header("Role & Parameters")
    role = st.selectbox("User role", ["member", "trainer", "manager", "admin", "all"], index=4)
    
    # Parameters for different roles
    member_id = st.selectbox("member_id", ["M_1001", "M_1002", "M_1003", "M_1004"])
    trainer_id = st.selectbox("trainer_id", ["T_1001", "T_1002", "T_1003"])
    target_sessions = st.number_input("target_sessions", min_value=1, value=10, step=1)
    days = st.slider("last N days", 1, 90, 7)

    PARAMS_CTX = {
        "member_id": member_id,
        "trainer_id": trainer_id,
        "target_sessions": int(target_sessions),
        "days": int(days),
    }

# Postgres part of the dashboard
st.subheader("Postgres")
try:
    eng = get_pg_engine(pg_uri)

    with st.expander("Run Postgres query", expanded=True):
        def filter_queries_by_role(qdict: dict, role: str) -> dict:
            def ok(tags):
                t = [s.lower() for s in (tags or ["all"])]
                return "all" in t or role.lower() in t
            return {name: q for name, q in qdict.items() if ok(q.get("tags"))}

        pg_all = CONFIG["postgres"]["queries"]
        pg_q = filter_queries_by_role(pg_all, role)

        names = list(pg_q.keys()) or ["(no queries for this role)"]
        sel = st.selectbox("Choose a saved query", names, key="pg_sel")

        if sel in pg_q:
            q = pg_q[sel]
            sql = qualify(q["sql"])   
            st.code(sql, language="sql")

            run  = auto_run or st.button("▶ Run Postgres", key="pg_run")
            if run:
                wanted = q.get("params", [])
                params = {k: PARAMS_CTX[k] for k in wanted}
                df = run_pg_query(eng, sql, params=params)
                render_chart(df, q["chart"])
        else:
            st.info("No Postgres queries tagged for this role.")
except Exception as e:
    st.error(f"Postgres error: {e}")

# Mongo panel
if CONFIG["mongo"]["enabled"]:
    st.subheader("🍃 MongoDB")
    try:
        mongo_client = get_mongo_client(mongo_uri)   
        metric_row(mongo_overview(mongo_client, mongo_db))

        with st.expander("Run Mongo aggregation", expanded=True):
            mongo_query_names = list(CONFIG["mongo"]["queries"].keys())
            selm = st.selectbox("Choose a saved aggregation", mongo_query_names, key="mongo_sel")
            q = CONFIG["mongo"]["queries"][selm]
            st.write(f"**Collection:** `{q['collection']}`")
            st.code(str(q["aggregate"]), language="python")
            runm = auto_run or st.button("▶ Run Mongo", key="mongo_run")
            if runm:
                dfm = run_mongo_aggregate(mongo_client, mongo_db, q["collection"], q["aggregate"])
                render_chart(dfm, q["chart"])
    except Exception as e:
        st.error(f"Mongo error: {e}")