import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const Graph = ({data}) => {
    if (data.length === 0) return null;

    return (
        <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
                dataKey="time" 
                ticks={Array.from({ length: Math.ceil(Math.max(...data.map(d => d.time)) / 0.5) }, (_, i) => i * 0.5)}
            />
            <YAxis />
            <Tooltip contentStyle={{ fontSize: "12px", padding: "2px 6px", lineHeight: "1" }} />
            <Line type="monotone" dataKey="force" stroke="#8884d8" dot={{ r: 1 }}/>
            </LineChart>
        </ResponsiveContainer>
    )
}

export default Graph